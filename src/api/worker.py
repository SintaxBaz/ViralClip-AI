"""Background worker pool.

The reason this file exists: the pipeline is blocking, CPU-bound work that
takes minutes. Two consequences shape the design.

1. It cannot run inside the request. An HTTP client, and every proxy
   between it and us, will time out long before a 40-minute video is
   transcribed. So POST /jobs enqueues and returns 202 immediately, and
   the client polls.

2. It cannot run on the event loop. `async def` does not make blocking
   work concurrent -- a synchronous Whisper call in a coroutine would
   freeze the loop and stall every other request, including the status
   polls. So the pipeline runs in a thread pool via run_in_executor, and
   the loop stays free to serve requests.

Concurrency is deliberately bounded (default 1). Model inference holds its
weights resident for the whole run, so raising it multiplies peak memory
rather than throughput. Scale with replicas, not threads.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from pipeline import ModelRegistry, PipelineError, Stage, run_pipeline

from .config import Settings
from .jobs import JobStore

log = logging.getLogger("viralclip.worker")


class QueueFull(RuntimeError):
    """The queue is at capacity; the caller should return 503."""


class JobWorker:
    def __init__(
        self,
        store: JobStore,
        settings: Settings,
        registry: ModelRegistry | None = None,
        pipeline_fn=run_pipeline,
    ):
        self.store = store
        self.settings = settings
        self.registry = registry
        # Injectable so the API can be tested without loading models.
        self.pipeline_fn = pipeline_fn
        self._queue: asyncio.Queue[str] = asyncio.Queue(
            maxsize=settings.max_queue_depth
        )
        self._executor = ThreadPoolExecutor(
            max_workers=settings.worker_concurrency,
            thread_name_prefix="viralclip",
        )
        self._tasks: list[asyncio.Task] = []
        self._started = False

    # ---- lifecycle ----

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        for i in range(self.settings.worker_concurrency):
            self._tasks.append(asyncio.create_task(self._run(i), name=f"worker-{i}"))
        log.info("Started %d worker(s)", self.settings.worker_concurrency)

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._started = False

    # ---- submission ----

    def submit(self, job_id: str) -> None:
        """Enqueue a job. Raises QueueFull if the queue is at capacity."""
        try:
            self._queue.put_nowait(job_id)
        except asyncio.QueueFull as exc:
            raise QueueFull(
                f"Queue is full ({self.settings.max_queue_depth} waiting)."
            ) from exc

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    # ---- the loop ----

    async def _run(self, index: int) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                await self._process(job_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Worker %d crashed on job %s", index, job_id)
            finally:
                self._queue.task_done()

    async def _process(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None:
            return

        self.store.update(
            job_id,
            stage=Stage.DOWNLOADING,
            percent=1,
            detail="Starting",
            started_at=datetime.now(timezone.utc),
        )

        # Called from the worker thread. Single attribute writes under the
        # store's lock, so no extra synchronization is needed here.
        def progress(stage: Stage, percent: int, detail: str) -> None:
            self.store.update(job_id, stage=stage, percent=percent, detail=detail)

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                self._executor,
                lambda: self.pipeline_fn(
                    query=job.query,
                    url=job.source_ref if job.source_kind == "url" else None,
                    file=job.source_ref if job.source_kind == "upload" else None,
                    output_dir=job.output_dir,
                    progress=progress,
                    registry=self.registry,
                    **job.params,
                ),
            )
        except PipelineError as exc:
            log.warning("Job %s failed: %s", job_id, exc)
            self.store.update(
                job_id,
                stage=Stage.FAILED,
                error=str(exc),
                detail="Failed",
                finished_at=datetime.now(timezone.utc),
            )
            return
        except Exception as exc:  # noqa: BLE001
            # Never leak an internal traceback to an API caller.
            log.exception("Job %s raised an unexpected error", job_id)
            self.store.update(
                job_id,
                stage=Stage.FAILED,
                error=f"Internal error: {type(exc).__name__}",
                detail="Failed",
                finished_at=datetime.now(timezone.utc),
            )
            return

        self.store.update(
            job_id,
            stage=Stage.DONE,
            percent=100,
            detail=f"{len(result.clips)} clip(s) ready",
            result=result,
            finished_at=datetime.now(timezone.utc),
        )
