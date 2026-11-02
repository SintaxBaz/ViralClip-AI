"""Real ffmpeg integration test.

Generates a synthetic video with ffmpeg, cuts clips from it, and verifies
the output durations with ffprobe. Unlike the other suites this touches
real binaries -- it SKIPS cleanly when ffmpeg is unavailable, so it is
safe to run anywhere and still does real work in CI.

    python tests/test_ffmpeg_integration.py
"""

import subprocess
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

for name in ("whisper", "pyannote", "pyannote.audio", "sentence_transformers",
             "moviepy", "dotenv", "yt_dlp", "torch"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["moviepy"].VideoFileClip = object
sys.modules["dotenv"].load_dotenv = lambda *a, **k: None
sys.modules["sentence_transformers"].SentenceTransformer = object
sys.modules["sentence_transformers"].util = types.SimpleNamespace(semantic_search=None)

from video_editor import VideoEditor, clip_stem, ffmpeg_available, resolve_window  # noqa: E402

if not ffmpeg_available():
    print("SKIP: ffmpeg not on PATH; nothing to integrate against.")
    sys.exit(0)

passed = failed = 0


def check(label, got, want=True):
    global passed, failed
    if got == want:
        passed += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}\n          got:  {got!r}\n          want: {want!r}")


def duration(path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


with tempfile.TemporaryDirectory(prefix="vc_ffmpeg_") as tmp:
    tmp = Path(tmp)
    source = tmp / "source.mp4"

    print("generating a 60s synthetic source")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=duration=60:size=320x240:rate=25",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=60",
         "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source)],
        check=True,
    )
    check("source is 60s", abs(duration(source) - 60.0) < 0.5)

    editor = VideoEditor(output_dir=str(tmp / "clips"))

    print("\ncutting clips at resolved windows")
    for raw, label in [((5.0, 8.0), "short match padded to the 15s floor"),
                       ((20.0, 40.0), "long match truncated to 25s ceiling"),
                       ((0.2, 3.0), "match at the very start")]:
        start, end = resolve_window(*raw)
        path = Path(editor.cut_clip(str(source), start, end))
        check(f"{label}: file written", path.is_file())
        check(f"{label}: duration matches window",
              abs(duration(path) - (end - start)) < 0.5)
        check(f"{label}: named from the window",
              path.stem, clip_stem(start, end))

    print("\nclip is a real, playable mp4")
    start, end = resolve_window(10.0, 20.0)
    path = Path(editor.cut_clip(str(source), start, end))
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    check("video stream is h264 at source size", probe, "h264,320,240")
    audio = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    check("audio stream is aac", audio, "aac")

    print("\nfailure path")
    try:
        editor.cut_clip(str(tmp / "absent.mp4"), 0.0, 15.0)
        check("missing input raises", False)
    except RuntimeError as exc:
        check("missing input raises RuntimeError", True)
        check("error carries ffmpeg's own message", "ffmpeg failed" in str(exc))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
