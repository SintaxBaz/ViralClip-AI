"""Dependency-free smoke tests.

Whisper, pyannote, moviepy and sentence-transformers are stubbed, so this
runs in under a second without downloading ~2 GB of model weights. It covers
the pure logic: clip boundary maths, clip/subtitle name pairing, SRT output
and CLI argument handling.

    python tests/test_smoke.py
"""

import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# --- stub heavy third-party imports before importing our modules ----------
for name in ("whisper", "pyannote", "pyannote.audio", "sentence_transformers",
             "moviepy", "dotenv", "yt_dlp", "torch"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["moviepy"].VideoFileClip = object
sys.modules["dotenv"].load_dotenv = lambda *a, **k: None
sys.modules["sentence_transformers"].SentenceTransformer = object
sys.modules["sentence_transformers"].util = types.SimpleNamespace(semantic_search=None)
sys.modules["pyannote.audio"].Pipeline = object

from subtitle_agent import ReelSubtitleAgent          # noqa: E402
from video_editor import clip_stem, resolve_window     # noqa: E402

passed = failed = 0


def check(label, got, want):
    global passed, failed
    if got == want:
        passed += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}\n          got:  {got!r}\n          want: {want!r}")


print("resolve_window: padding and clamping")
# 3s match, 2s padding -> 1..7 = 6s, below the 15s floor, so extended to 15s.
check("short match extends to min_len", resolve_window(3.0, 6.0), (1.0, 16.0))
# A 40s match padded to 44s exceeds the 25s ceiling, so truncated to 25s.
check("long match truncates to max_len", resolve_window(10.0, 50.0), (8.0, 33.0))
# Padding must never push the start before zero.
check("start clamps at zero", resolve_window(0.5, 4.0), (0.0, 15.0))
# A match already inside the bounds is padded and left alone.
check("in-range match untouched", resolve_window(30.0, 48.0), (28.0, 50.0))
check("custom bounds respected",
      resolve_window(10.0, 12.0, padding=1.0, min_len=5.0, max_len=8.0), (9.0, 14.0))

print("\nclip_stem: clips and subtitles must pair up")
# This is the regression that previously produced clip_13_28.mp4 beside
# sub_14.0_29.0.srt -- two names for one window.
start, end = resolve_window(13.0, 24.0)
stem = clip_stem(start, end)
check("one window yields one stem", stem, "clip_11_26")
check("clip and srt share a stem", (f"{stem}.mp4", f"{stem}.srt"),
      ("clip_11_26.mp4", "clip_11_26.srt"))
check("stem is stable across calls", clip_stem(*resolve_window(13.0, 24.0)), stem)

print("\nReelSubtitleAgent: timestamps rebase to clip start")
agent = ReelSubtitleAgent(output_dir="/tmp/vc_test_subs")
check("timestamp format", agent.format_timestamp(3661.5), "01:01:01,500")
check("zero timestamp", agent.format_timestamp(0), "00:00:00,000")

segments = [
    {"start": 0.0,  "end": 5.0,  "text": "before the window"},
    {"start": 12.0, "end": 15.0, "text": "inside one"},
    {"start": 16.0, "end": 19.0, "text": "inside two"},
    {"start": 95.0, "end": 99.0, "text": "after the window"},
]
out = agent.save_srt(segments, 10.0, 25.0, output_name="pairing.srt")
body = Path(out).read_text(encoding="utf-8")
check("only in-window segments kept", body.count(" --> "), 2)
check("excludes earlier segment", "before the window" in body, False)
check("excludes later segment", "after the window" in body, False)
# 12.0s absolute, in a clip starting at 10.0s, must become 00:00:02,000.
check("first cue rebased to clip start", "00:00:02,000 --> 00:00:05,000" in body, True)

print("\nCLI: argument parsing and validation")
import main  # noqa: E402

parser = main.build_parser()
args = parser.parse_args(["--file", "x.mp4", "-q", "rockets"])
check("defaults", (args.top_k, args.model, args.no_diarize, args.dry_run),
      (5, "base", False, False))
args = parser.parse_args(["--url", "http://e.com/v", "-q", "a b", "-k", "3",
                          "--model", "small", "--no-diarize", "--dry-run"])
check("flags parse", (args.top_k, args.model, args.no_diarize, args.dry_run),
      (3, "small", True, True))


def exits_with(argv, code):
    try:
        return main.main(argv) == code
    except SystemExit as e:
        return e.code == code


check("--url and --file are mutually exclusive",
      exits_with(["--url", "u", "--file", "f", "-q", "x"], 2), True)
check("a source is required", exits_with(["-q", "x"], 2), True)
check("--query is required", exits_with(["--file", "x.mp4"], 2), True)
check("invalid --model rejected",
      exits_with(["--file", "x.mp4", "-q", "a", "--model", "huge"], 2), True)
check("min_len > max_len rejected",
      exits_with(["--file", "x.mp4", "-q", "a", "--min-len", "40", "--max-len", "10"], 2), True)
check("missing input file rejected",
      exits_with(["--file", "definitely_absent.mp4", "-q", "a", "--dry-run"], 2), True)

print("\nVideoDownloader: format selector must not silently drop audio")
import yt_dlp as _yt_dlp_stub  # the stub module installed above
_yt_dlp_stub.YoutubeDL = type("FakeYDL", (), {
    "__init__": lambda self, opts: setattr(self, "opts", opts),
    "__enter__": lambda self: self,
    "__exit__": lambda self, *a: None,
})
from video_downloader import VideoDownloader

captured = {}
class CapturingYDL(_yt_dlp_stub.YoutubeDL):
    def __init__(self, opts):
        super().__init__(opts)
        captured["opts"] = opts
_yt_dlp_stub.YoutubeDL = CapturingYDL
try:
    VideoDownloader(download_path="/tmp/vc_dl_test").download("http://example.com/v")
except Exception:
    pass  # extract_info isn't implemented on the fake; we only need opts
fmt = captured.get("opts", {}).get("format", "")
# The regression this guards: "mp4/best" picks a single mp4 stream, which
# on many YouTube videos is video-only (audio ships separately as m4a).
check("format requests audio+video explicitly, not a single mp4 stream",
      fmt != "mp4/best", True)
check("format pairs mp4 video with m4a audio",
      "bestaudio" in fmt and "bestvideo" in fmt, True)
check("merge_output_format set, so split streams are actually joined",
      captured.get("opts", {}).get("merge_output_format"), "mp4")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
