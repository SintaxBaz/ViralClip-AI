"""HTTP API for ViralClip-AI.

    uvicorn api.app:app --app-dir src --reload

Submit a job with either a URL or a file upload, then poll it:

    POST   /jobs              {"url": "...", "query": "..."}   -> 202 {id}
    POST   /jobs/upload       multipart: file=@video.mp4, query=...
    GET    /jobs/{id}                                          -> status
    GET    /jobs/{id}/clips                                    -> clip list
    GET    /jobs/{id}/clips/{name}                             -> the file
    GET    /jobs                                               -> recent jobs
    GET    /healthz                                            -> readiness
"""

from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from pipeline import ModelRegistry, Stage
from video_editor import ffmpeg_available

from .config import Settings
from .jobs import JobStore
from .worker import JobWorker, QueueFull

log = logging.getLogger("viralclip.api")

SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


class JobParams(BaseModel):
    """Tunables shared by both submission routes."""

    query: str = Field(min_length=1, max_length=200)
    top_k: int = Field(default=5, ge=1, le=20)
    padding: float = Field(default=2.0, ge=0.0, le=30.0)
    min_len: float = Field(default=15.0, gt=0.0, le=600.0)
    max_len: float = Field(default=25.0, gt=0.0, le=600.0)
    dry_run: bool = False

    def pipeline_kwargs(self, settings: Settings) -> dict:
        return {
            "top_k": self.top_k,
            "padding": self.padding,
            "min_len": self.min_len,
            "max_len": self.max_len,
            "dry_run": self.dry_run,
            "whisper_model": settings.whisper_model,
            "diarize": settings.diarize,
        }


class UrlJobRequest(JobParams):
    url: str = Field(min_length=1, max_length=2048)


def create_app(
    settings: Settings | None = None,
    pipeline_fn=None,
    warm: bool | None = None,
) -> FastAPI:
    """
    Build the app.

    `pipeline_fn` and `warm` exist so tests can run the full HTTP surface
    against a fake pipeline, without loading models.
    """
    settings = settings or Settings.from_env()
    store = JobStore()

    registry = ModelRegistry() if pipeline_fn is None else None
    should_warm = settings.warm_models if warm is None else warm

    worker_kwargs = {"registry": registry}
    if pipeline_fn is not None:
        worker_kwargs["pipeline_fn"] = pipeline_fn
    worker = JobWorker(store, settings, **worker_kwargs)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load models once, at startup, instead of on the first request.
        # Without this the first caller waits for a ~150 MB Whisper load
        # plus a ~90 MB embedder on top of their own job.
        if should_warm and registry is not None:
            log.info("Warming models (%s)", settings.whisper_model)
            registry.warm(settings.whisper_model)
            log.info("Models ready")
        await worker.start()
        yield
        await worker.stop()

    app = FastAPI(
        title="ViralClip-AI",
        version="0.2.0",
        description="Turn a long video into short, subtitled clips.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.store = store
    app.state.worker = worker

    # ---- helpers ----

    def enqueue(source_kind: str, source_ref: str, params: JobParams):
        if params.min_len > params.max_len:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"min_len ({params.min_len}) exceeds max_len ({params.max_len}).",
            )

        job = store.create(
            query=params.query,
            params=params.pipeline_kwargs(settings),
            source_kind=source_kind,
            source_ref=source_ref,
            output_dir=settings.output_root / "pending",
        )
        # One directory per job, so concurrent jobs cannot overwrite each
        # other's clips (they are named by timestamp, which collides).
        job_dir = settings.output_root / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        store.update(job.id, output_dir=job_dir)

        try:
            worker.submit(job.id)
        except QueueFull as exc:
            store.update(job.id, stage=Stage.FAILED, error=str(exc), detail="Rejected")
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

        return store.get(job.id)

    def require_job(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No such job: {job_id}")
        return job

    # ---- routes ----

    @app.get("/healthz")
    def healthz():
        """Readiness. Reports ffmpeg, because without it no job can finish."""
        ffmpeg = ffmpeg_available()
        body = {
            "status": "ok" if ffmpeg else "degraded",
            "ffmpeg": ffmpeg,
            "queue_depth": worker.queue_depth,
            "active_jobs": store.count_active(),
            "concurrency": settings.worker_concurrency,
            "whisper_model": settings.whisper_model,
            "diarize": settings.diarize,
        }
        code = status.HTTP_200_OK if ffmpeg else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(body, status_code=code)

    @app.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
    def submit_url(req: UrlJobRequest):
        """Submit a job from a video URL. Returns immediately; poll the job."""
        if not req.url.startswith(("http://", "https://")):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "url must start with http:// or https://",
            )
        job = enqueue("url", req.url, req)
        return job.as_dict()

    @app.post("/jobs/upload", status_code=status.HTTP_202_ACCEPTED)
    async def submit_upload(
        file: UploadFile = File(...),
        query: str = Form(...),
        top_k: int = Form(5),
        padding: float = Form(2.0),
        min_len: float = Form(15.0),
        max_len: float = Form(25.0),
        dry_run: bool = Form(False),
    ):
        """Submit a job from an uploaded video file."""
        params = JobParams(
            query=query, top_k=top_k, padding=padding,
            min_len=min_len, max_len=max_len, dry_run=dry_run,
        )

        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in settings.allowed_upload_suffixes:
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                f"Unsupported type {suffix or '(none)'}. "
                f"Allowed: {', '.join(settings.allowed_upload_suffixes)}",
            )

        incoming = settings.output_root / "uploads"
        incoming.mkdir(parents=True, exist_ok=True)
        safe_stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(file.filename).stem)[:80]
        target = incoming / f"{safe_stem}{suffix}"
        counter = 1
        while target.exists():
            target = incoming / f"{safe_stem}_{counter}{suffix}"
            counter += 1

        # Stream to disk in chunks and enforce the cap as we go, so an
        # oversized body is rejected mid-transfer rather than buffered
        # whole in memory first.
        written = 0
        try:
            with target.open("wb") as out:
                while chunk := await file.read(settings.upload_chunk_bytes):
                    written += len(chunk)
                    if written > settings.max_upload_bytes:
                        out.close()
                        target.unlink(missing_ok=True)
                        raise HTTPException(
                            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"Upload exceeds {settings.max_upload_bytes} bytes.",
                        )
                    out.write(chunk)
        except HTTPException:
            raise
        except Exception as exc:
            target.unlink(missing_ok=True)
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to store upload."
            ) from exc

        if written == 0:
            target.unlink(missing_ok=True)
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, "Uploaded file is empty."
            )

        job = enqueue("upload", str(target), params)
        return job.as_dict()

    @app.get("/jobs")
    def list_jobs(limit: int = 50):
        limit = max(1, min(limit, 200))
        return {"jobs": [j.as_dict() for j in store.list(limit)]}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        return require_job(job_id).as_dict()

    @app.get("/jobs/{job_id}/clips")
    def list_clips(job_id: str, request: Request):
        job = require_job(job_id)
        if not job.is_terminal:
            return {"status": job.stage.value, "percent": job.percent, "clips": []}
        if job.stage is Stage.FAILED:
            raise HTTPException(status.HTTP_409_CONFLICT, job.error or "Job failed.")

        base = str(request.base_url).rstrip("/")
        clips = []
        for clip in (job.result.clips if job.result else []):
            body = clip.as_dict()
            if body["clip"]:
                body["download"] = f"{base}/jobs/{job_id}/clips/{body['clip']}"
            if body["subtitle"]:
                body["subtitle_download"] = (
                    f"{base}/jobs/{job_id}/clips/{body['subtitle']}"
                )
            clips.append(body)
        return {"status": job.stage.value, "clips": clips}

    @app.get("/jobs/{job_id}/clips/{name}")
    def download_clip(job_id: str, name: str):
        job = require_job(job_id)

        # Reject anything that is not a bare filename before it reaches the
        # filesystem, so ../ cannot escape the job directory.
        if not SAFE_NAME.match(name):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid filename.")

        subdir = "subtitles" if name.endswith(".srt") else "clips"
        path = (job.output_dir / subdir / name).resolve()

        # Belt and braces: confirm the resolved path is still inside the job.
        if not str(path).startswith(str(job.output_dir.resolve())):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid path.")
        if not path.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No such file: {name}")

        media = "application/x-subrip" if name.endswith(".srt") else "video/mp4"
        return FileResponse(path, media_type=media, filename=name)

    return app


app = create_app()
