"""Audio extraction, transcription (Whisper) and optional speaker diarization."""

import os
from pathlib import Path

import whisper
from dotenv import load_dotenv
from moviepy import VideoFileClip

# Project root = parent of src/. Used so paths work regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_hf_token() -> str | None:
    """
    Find a HuggingFace token.

    Checks, in order:
      1. HF_TOKEN already in the environment
      2. <project root>/.env
      3. <project root>/.env/HF_TOKEN.ini   (the original layout)

    Returns None if no token is found, which is not an error -- the caller
    falls back to transcription without diarization.
    """
    if os.getenv("HF_TOKEN"):
        return os.getenv("HF_TOKEN")

    for candidate in (PROJECT_ROOT / ".env", PROJECT_ROOT / ".env" / "HF_TOKEN.ini"):
        if candidate.is_file():
            load_dotenv(dotenv_path=candidate)
            if os.getenv("HF_TOKEN"):
                return os.getenv("HF_TOKEN")

    return None


def sanitize_filename(path: str) -> str:
    """
    Replace spaces in a filename with underscores and return the new path.

    Spaces in paths break naive shell quoting downstream, so the pipeline
    normalizes once up front.
    """
    directory, filename = os.path.split(path)
    if " " not in filename:
        return path

    safe_path = os.path.join(directory, filename.replace(" ", "_"))
    if os.path.exists(safe_path):
        # Target already exists -- drop the redundant spaced copy.
        if os.path.abspath(safe_path) != os.path.abspath(path):
            os.remove(path)
    else:
        os.rename(path, safe_path)
    return safe_path


def extract_audio(video_path: str, audio_path: str) -> None:
    """Extract the audio track of a video to a .wav file."""
    with VideoFileClip(video_path) as clip:
        if clip.audio is None:
            raise RuntimeError(f"No audio track found in {video_path}")
        clip.audio.write_audiofile(audio_path, logger=None)


def _whisper_segments(result: dict) -> list[dict]:
    """Whisper's own segments, shaped like the diarized output but with no speaker."""
    return [
        {
            "speaker": None,
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        }
        for seg in result["segments"]
        if seg["text"].strip()
    ]


def _diarized_segments(result: dict, audio_path: str, hf_token: str) -> list[dict]:
    """
    Re-cut Whisper's word timings along pyannote speaker turns.

    For each diarization turn, collect every Whisper word whose timing overlaps
    that turn, and emit one segment per turn.
    """
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    diarization = pipeline(audio_path)

    # Flatten all words once, instead of re-scanning every segment per turn.
    words = [
        word
        for segment in result["segments"]
        for word in segment.get("words", [])
    ]

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        in_turn = [
            w["word"].strip()
            for w in words
            if not (w["end"] < turn.start or w["start"] > turn.end)
        ]
        if in_turn:
            segments.append(
                {
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end,
                    "text": " ".join(in_turn),
                }
            )
    return segments


def transcribe_video(
    video_path: str,
    model_name: str = "base",
    diarize: bool = True,
) -> list[dict]:
    """
    Transcribe a video, optionally splitting the transcript by speaker.

    Returns a list of {speaker, start, end, text}. `speaker` is None when
    diarization is skipped.
    """
    video_path = sanitize_filename(video_path)
    audio_path = video_path + ".wav"
    extract_audio(video_path, audio_path)

    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, word_timestamps=True)

    if not diarize:
        return _whisper_segments(result)

    hf_token = load_hf_token()
    if not hf_token:
        print(
            "  ! No HF_TOKEN found -- continuing without speaker diarization.\n"
            "    See README.md to enable it, or pass --no-diarize to silence this."
        )
        return _whisper_segments(result)

    try:
        return _diarized_segments(result, audio_path, hf_token)
    except Exception as exc:  # noqa: BLE001 - diarization is strictly optional
        print(f"  ! Diarization failed ({exc}) -- falling back to plain transcript.")
        return _whisper_segments(result)


def save_transcription(transcription_segments: list[dict], output_path: str) -> None:
    """Write the transcript with timestamps (and speaker labels, if present)."""
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in transcription_segments:
            speaker = segment.get("speaker")
            label = f" {speaker}:" if speaker else ""
            f.write(
                f"[{segment['start']:.2f} - {segment['end']:.2f}]{label} {segment['text']}\n"
            )
