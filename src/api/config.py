"""Service configuration, read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Where job output is written. One subdirectory per job.
    output_root: Path = PROJECT_ROOT / "output" / "jobs"

    # How many videos may be processed at once.
    #
    # Deliberately 1 by default. Whisper inference is CPU/GPU bound and
    # holds its weights in memory for the whole run, so raising this
    # multiplies peak memory rather than throughput. Scale out with more
    # replicas, not more threads per replica.
    worker_concurrency: int = 1

    # Queue depth. Beyond this, submissions are rejected with 503 rather
    # than accepted into a queue nobody will get to.
    max_queue_depth: int = 32

    # Reject uploads larger than this. Enforced while streaming, so a
    # client cannot force us to buffer an unbounded body.
    max_upload_bytes: int = 2 * 1024 * 1024 * 1024  # 2 GiB

    upload_chunk_bytes: int = 1024 * 1024

    allowed_upload_suffixes: tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")

    whisper_model: str = "base"

    # Off by default: diarization needs a HuggingFace token and acceptance
    # of two gated models, which is a poor default for a service.
    diarize: bool = False

    # Load models at startup rather than on first request.
    warm_models: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        root = os.environ.get("VIRALCLIP_OUTPUT_ROOT")
        return cls(
            output_root=Path(root) if root else cls.output_root,
            worker_concurrency=_int("VIRALCLIP_CONCURRENCY", cls.worker_concurrency),
            max_queue_depth=_int("VIRALCLIP_MAX_QUEUE", cls.max_queue_depth),
            max_upload_bytes=_int("VIRALCLIP_MAX_UPLOAD_BYTES", cls.max_upload_bytes),
            whisper_model=os.environ.get("VIRALCLIP_WHISPER_MODEL", cls.whisper_model),
            diarize=_bool("VIRALCLIP_DIARIZE", cls.diarize),
            warm_models=_bool("VIRALCLIP_WARM_MODELS", cls.warm_models),
        )
