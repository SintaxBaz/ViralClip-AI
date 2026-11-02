"""Job records and the in-memory job store."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pipeline import PipelineResult, Stage

TERMINAL = {Stage.DONE, Stage.FAILED}


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Job:
    id: str
    query: str
    params: dict
    source_kind: str                       # "url" or "upload"
    source_ref: str                        # the URL, or the uploaded filename
    output_dir: Path
    stage: Stage = Stage.QUEUED
    percent: int = 0
    detail: str = "Waiting for a worker"
    error: str | None = None
    result: PipelineResult | None = None
    created_at: datetime = field(default_factory=_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        return self.stage in TERMINAL

    def as_dict(self) -> dict:
        body = {
            "id": self.id,
            "status": self.stage.value,
            "percent": self.percent,
            "detail": self.detail,
            "query": self.query,
            "source": {"kind": self.source_kind, "ref": self.source_ref},
            "params": self.params,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }
        if self.error:
            body["error"] = self.error
        if self.result is not None:
            body["result"] = self.result.as_dict()
        return body


class JobStore:
    """
    In-memory job registry.

    Guarded by a plain threading.Lock rather than an asyncio lock because
    the worker mutates jobs from a thread-pool thread while the event loop
    reads them to answer GET /jobs/{id}.

    Known limitation, and the first thing to change before this is more
    than a demo: jobs live in process memory, so a restart loses them and
    a second replica cannot see them. Moving this to Redis or Postgres is
    a drop-in swap behind these five methods -- which is why the API never
    touches the dict directly.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, **kwargs) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], **kwargs)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)

    def count_active(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if not j.is_terminal)
