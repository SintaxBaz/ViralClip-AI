"""The clip-extraction pipeline, independent of how it was invoked.

Both the CLI (main.py) and the HTTP API (api/) call run_pipeline(). Keeping
one implementation means the service and the command line cannot drift.

Two things make this reusable by a long-running service rather than only a
short-lived process:

  * ModelRegistry caches loaded models, so a service pays the model load
    cost once at startup instead of on every request.
  * run_pipeline() reports progress through a callback instead of printing,
    so the caller decides whether that becomes stdout or a job record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from clip_finder import ClipFinder
from subtitle_agent import ReelSubtitleAgent
from transcriber import save_transcription, transcribe_video
from video_downloader import VideoDownloader
from video_editor import VideoEditor, clip_stem, resolve_window


class Stage(str, Enum):
    """Pipeline stages, also used as the job status in the API."""

    QUEUED = "queued"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    SEARCHING = "searching"
    CUTTING = "cutting"
    SUBTITLING = "subtitling"
    DONE = "done"
    FAILED = "failed"


# (stage, percent complete, human-readable detail)
ProgressFn = Callable[[Stage, int, str], None]


def _noop(stage: Stage, percent: int, detail: str) -> None:
    pass


@dataclass
class ClipResult:
    stem: str
    start: float
    end: float
    score: float
    text: str
    clip_path: str | None = None
    subtitle_path: str | None = None

    def as_dict(self) -> dict:
        return {
            "stem": self.stem,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.end - self.start, 3),
            "score": round(self.score, 4),
            "text": self.text,
            "clip": Path(self.clip_path).name if self.clip_path else None,
            "subtitle": Path(self.subtitle_path).name if self.subtitle_path else None,
        }


@dataclass
class PipelineResult:
    source_path: str
    transcript_path: str
    segment_count: int
    clips: list[ClipResult] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "source": Path(self.source_path).name,
            "transcript": Path(self.transcript_path).name,
            "segments": self.segment_count,
            "clips": [c.as_dict() for c in self.clips],
        }


class PipelineError(RuntimeError):
    """A pipeline failure with a message safe to show an API caller."""


class ModelRegistry:
    """
    Caches loaded models across pipeline runs.

    The CLI leaves this empty (one run, then exit). The API creates one at
    startup and warms it, so a request never pays the load cost. Whisper
    'base' is ~150 MB and the MiniLM embedder ~90 MB; reloading them per
    request would dominate the response time for short videos.
    """

    def __init__(self, embedder_name: str = "all-MiniLM-L6-v2"):
        self.embedder_name = embedder_name
        self._embedder = None
        self._whisper: dict[str, object] = {}

    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embedder_name)
        return self._embedder

    def whisper(self, model_name: str):
        if model_name not in self._whisper:
            import whisper

            self._whisper[model_name] = whisper.load_model(model_name)
        return self._whisper[model_name]

    def warm(self, whisper_model: str = "base") -> None:
        """Load eagerly, at startup, so the first request is not the slow one."""
        self.embedder()
        self.whisper(whisper_model)


def run_pipeline(
    *,
    query: str,
    url: str | None = None,
    file: str | None = None,
    output_dir: str | Path,
    top_k: int = 5,
    whisper_model: str = "base",
    diarize: bool = True,
    padding: float = 2.0,
    min_len: float = 15.0,
    max_len: float = 25.0,
    dry_run: bool = False,
    progress: ProgressFn | None = None,
    registry: ModelRegistry | None = None,
) -> PipelineResult:
    """
    Run the full pipeline. Exactly one of `url` or `file` must be given.

    Raises PipelineError with a caller-safe message on any failure.
    """
    if bool(url) == bool(file):
        raise PipelineError("Provide exactly one of url or file.")
    if min_len > max_len:
        raise PipelineError(f"min_len ({min_len}) exceeds max_len ({max_len}).")
    if top_k < 1:
        raise PipelineError("top_k must be at least 1.")

    report = progress or _noop
    out = Path(output_dir)
    for sub in ("downloads", "clips", "subtitles"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # ---- 1. source ----
    if file:
        source = Path(file)
        if not source.is_file():
            raise PipelineError(f"No such file: {source}")
        video_path = str(source)
        report(Stage.DOWNLOADING, 5, f"Using local file {source.name}")
    else:
        report(Stage.DOWNLOADING, 5, "Downloading source video")
        try:
            video_path = VideoDownloader(
                download_path=str(out / "downloads")
            ).download(url)
        except Exception as exc:
            raise PipelineError(f"Download failed: {exc}") from exc
        report(Stage.DOWNLOADING, 15, f"Downloaded {Path(video_path).name}")

    # ---- 2. transcribe ----
    report(Stage.TRANSCRIBING, 20, f"Transcribing with Whisper '{whisper_model}'")
    try:
        segments = transcribe_video(
            video_path,
            model_name=whisper_model,
            diarize=diarize,
            model=registry.whisper(whisper_model) if registry else None,
        )
    except Exception as exc:
        raise PipelineError(f"Transcription failed: {exc}") from exc

    if not segments:
        raise PipelineError("Transcription produced no segments (is there audio?).")

    transcript_path = out / f"{Path(video_path).stem}_transcription.txt"
    save_transcription(segments, str(transcript_path))
    report(Stage.TRANSCRIBING, 60, f"{len(segments)} transcript segments")

    result = PipelineResult(
        source_path=video_path,
        transcript_path=str(transcript_path),
        segment_count=len(segments),
    )

    # ---- 3. search ----
    report(Stage.SEARCHING, 65, f"Searching for {query!r}")
    finder = ClipFinder(embedder=registry.embedder() if registry else None)
    matches = finder.search_segments(segments, query, top_k=top_k)
    if not matches:
        report(Stage.DONE, 100, f"No matches for {query!r}")
        return result

    # ---- 4. windows ----
    for match in matches:
        start, end = resolve_window(
            match["start"], match["end"], padding, min_len, max_len
        )
        result.clips.append(
            ClipResult(
                stem=clip_stem(start, end),
                start=start,
                end=end,
                score=match["score"],
                text=match["text"],
            )
        )

    if dry_run:
        report(Stage.DONE, 100, f"{len(result.clips)} window(s), nothing written")
        return result

    # ---- 5. cut + subtitle ----
    editor = VideoEditor(output_dir=str(out / "clips"))
    agent = ReelSubtitleAgent(output_dir=str(out / "subtitles"))
    total = len(result.clips)

    for i, clip in enumerate(result.clips, 1):
        report(
            Stage.CUTTING,
            65 + int(30 * (i - 1) / total),
            f"Cutting clip {i} of {total}",
        )
        try:
            clip.clip_path = editor.cut_clip(
                result.source_path, clip.start, clip.end, output_name=clip.stem
            )
        except Exception as exc:
            raise PipelineError(f"Cutting {clip.stem} failed: {exc}") from exc

        clip.subtitle_path = agent.save_srt(
            segments, clip.start, clip.end, output_name=f"{clip.stem}.srt"
        )

    report(Stage.SUBTITLING, 98, f"{total} clip(s) written")
    report(Stage.DONE, 100, f"{total} clip(s) ready")
    return result
