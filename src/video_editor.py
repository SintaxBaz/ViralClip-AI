"""Clip cutting via ffmpeg."""

import os
import shutil
import subprocess
from typing import List, Tuple

DEFAULT_PADDING = 2.0
DEFAULT_MIN_LEN = 15.0
DEFAULT_MAX_LEN = 25.0


def ffmpeg_available() -> bool:
    """True if ffmpeg is on PATH."""
    return shutil.which("ffmpeg") is not None


def resolve_window(
    start: float,
    end: float,
    padding: float = DEFAULT_PADDING,
    min_len: float = DEFAULT_MIN_LEN,
    max_len: float = DEFAULT_MAX_LEN,
) -> Tuple[float, float]:
    """
    Turn a raw transcript match into the actual clip window.

    Pads both sides, then clamps the result into [min_len, max_len].

    This is the single source of truth for clip boundaries. It previously
    lived in two places -- here and in main.py -- which disagreed (int() vs
    round()), so generated clips and their .srt files were named with
    different timestamps and could not be paired up.
    """
    start_time = max(0.0, start - padding)
    end_time = end + padding

    duration = end_time - start_time
    if duration < min_len:
        end_time = start_time + min_len
    elif duration > max_len:
        end_time = start_time + max_len

    return start_time, end_time


def clip_stem(start_time: float, end_time: float) -> str:
    """Canonical basename for a clip window, shared by clips and subtitles."""
    return f"clip_{int(start_time)}_{int(end_time)}"


class VideoEditor:
    def __init__(self, output_dir: str = "clips"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def cut_clip(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_name: str | None = None,
    ) -> str:
        """
        Cut [start_time, end_time] out of video_path. Window must already be
        resolved by resolve_window().
        """
        duration = end_time - start_time
        name = output_name or clip_stem(start_time, end_time)
        filename = os.path.join(self.output_dir, f"{name}.mp4")

        command = [
            "ffmpeg", "-y",
            # -ss BEFORE -i: ffmpeg seeks to the keyframe first instead of
            # decoding the whole file from 0. Still frame-accurate because
            # we re-encode. This is the difference between ~1s and ~30s
            # per clip on a long source.
            "-ss", f"{start_time:.3f}",
            "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-c:a", "aac",
            filename,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            tail = result.stderr.strip().splitlines()[-3:]
            raise RuntimeError(
                f"ffmpeg failed on {name}:\n  " + "\n  ".join(tail)
            )
        return filename

    def batch_cut_clips(
        self, video_path: str, windows: List[Tuple[float, float]]
    ) -> List[str]:
        """Cut each already-resolved (start, end) window."""
        return [self.cut_clip(video_path, s, e) for s, e in windows]
