"""Diarization must be genuinely optional.

The default container does not install pyannote.audio, and
requirements.txt leaves it out, so this is the test that keeps that safe.
If someone moves the pyannote import to module scope, this fails.

    python tests/test_optional_deps.py
"""

import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# Stub everything EXCEPT pyannote, reproducing the default image.
for name in ("whisper", "moviepy", "dotenv", "yt_dlp", "torch"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["moviepy"].VideoFileClip = object
sys.modules["dotenv"].load_dotenv = lambda *a, **k: None

import transcriber  # noqa: E402

passed = failed = 0


def check(label, got, want=True):
    global passed, failed
    if got == want:
        passed += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}\n          got:  {got!r}\n          want: {want!r}")


class FakeWhisper:
    def transcribe(self, *a, **k):
        return {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": " hello there",
                 "words": [{"word": " hello", "start": 0.0, "end": 1.0},
                           {"word": " there", "start": 1.0, "end": 2.0}]},
                {"start": 2.0, "end": 4.0, "text": " second line",
                 "words": [{"word": " second", "start": 2.0, "end": 3.0},
                           {"word": " line", "start": 3.0, "end": 4.0}]},
            ]
        }


transcriber.extract_audio = lambda v, a: None
transcriber.sanitize_filename = lambda p: p

check("pyannote is not imported at module load", "pyannote.audio" not in sys.modules)

print("\nwith no HuggingFace token")
transcriber.load_hf_token = lambda: None
segs = transcriber.transcribe_video("v.mp4", model=FakeWhisper(), diarize=True)
check("returns segments", len(segs), 2)
check("speaker is None", segs[0]["speaker"], None)
check("text is preserved", segs[0]["text"], "hello there")

print("\nwith a token but pyannote not installed")
transcriber.load_hf_token = lambda: "hf_faketoken"
segs = transcriber.transcribe_video("v.mp4", model=FakeWhisper(), diarize=True)
check("degrades instead of raising ImportError", len(segs), 2)
check("speaker is None", segs[1]["speaker"], None)

print("\nwith diarization explicitly disabled")
segs = transcriber.transcribe_video("v.mp4", model=FakeWhisper(), diarize=False)
check("returns segments", len(segs), 2)
check("speaker is None", segs[0]["speaker"], None)

print("\nsave_transcription handles a speakerless transcript")
out = Path("/tmp/vc_optdeps.txt")
transcriber.save_transcription(segs, str(out))
body = out.read_text(encoding="utf-8")
check("writes both lines", body.count("\n"), 2)
check("omits the speaker label", "None:" not in body)
check("keeps timestamps", "[0.00 - 2.00]" in body)
out.unlink(missing_ok=True)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
