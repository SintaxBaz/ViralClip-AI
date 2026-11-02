# ViralClip-AI

Find the moments in a long video that match a search query, and cut them into
short, subtitled clips ready for Reels / Shorts / TikTok.

Give it a YouTube URL or a local file plus something to search for. It
transcribes the audio with Whisper, optionally splits the transcript by
speaker, finds the matching moments by **both keyword and meaning**, and cuts
each one to a 15–25 second clip with a matching `.srt`.

> **Scope:** clip selection is *query-driven* — you tell it what to look for.
> It does not yet rank moments by engagement on its own.

---

## How it works

```
  URL / local file
        |
   [1] yt-dlp ................ download source video
        |
   [2] moviepy ............... extract audio to .wav
       Whisper ............... transcribe with per-word timestamps
       pyannote.audio ........ (optional) label speaker turns
        |
   [3] sentence-transformers . hybrid search over transcript segments
        |                       exact substring match + MiniLM cosine similarity
   [4] resolve_window() ...... pad each match, clamp to 15-25s
       ffmpeg ................ cut the clip
        |
   [5] subtitle_agent ........ write .srt with timestamps rebased to 00:00
        |
   output/clips/clip_13_28.mp4
   output/subtitles/clip_13_28.srt
```

| Module | Responsibility |
|---|---|
| `src/video_downloader.py` | Fetch the source video via `yt-dlp` |
| `src/transcriber.py` | Audio extraction, Whisper transcription, optional diarization |
| `src/clip_finder.py` | Hybrid keyword + semantic transcript search |
| `src/video_editor.py` | Clip boundary maths and `ffmpeg` cutting |
| `src/subtitle_agent.py` | Per-clip `.srt` generation |
| `src/main.py` | CLI and pipeline orchestration |

---

## Requirements

- **Python 3.10+**
- **ffmpeg on your PATH** — not a pip package, install it separately:
  - Windows: `winget install Gyan.FFmpeg` (then reopen your terminal)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
  - Verify with `ffmpeg -version`
- ~2 GB free disk for model weights on first run
- A GPU is optional. On CPU, expect roughly real-time transcription with
  `--model base`; `--model large` will be far slower.

## Install

```bash
git clone https://github.com/SintaxBaz/ViralClip-AI.git
cd ViralClip-AI

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

For CUDA, install `torch` from the
[official selector](https://pytorch.org/get-started/locally/) *before*
running the line above.

## Usage

```bash
# From a YouTube URL
python src/main.py --url "https://www.youtube.com/watch?v=0dyEOsmX4iA" -q flamethrower

# From a local file
python src/main.py --file output/downloads/podcast.mp4 -q "space travel"

# Preview the clip windows without cutting anything (fast, no ffmpeg needed)
python src/main.py --file output/downloads/podcast.mp4 -q rockets --dry-run

# Skip diarization (no HuggingFace token required), cap at 3 clips
python src/main.py --file output/downloads/podcast.mp4 -q rockets --no-diarize -k 3
```

Run `python src/main.py --help` for the full list.

| Flag | Default | Meaning |
|---|---|---|
| `--url` / `--file` | — | Source video; exactly one is required |
| `-q`, `--query` | — | What to search the transcript for (required) |
| `-k`, `--top-k` | `5` | Maximum clips to produce |
| `--model` | `base` | Whisper size: `tiny`…`large` |
| `--no-diarize` | off | Skip speaker labelling |
| `--padding` | `2.0` | Seconds of context kept each side of a match |
| `--min-len` / `--max-len` | `15.0` / `25.0` | Clip length bounds |
| `-o`, `--output-dir` | `./output` | Where results are written |
| `--dry-run` | off | Print windows, write nothing |

Output layout:

```
output/
  downloads/    source videos + extracted .wav
  clips/        clip_<start>_<end>.mp4
  subtitles/    clip_<start>_<end>.srt   <- same stem as its clip
  *_transcription.txt
```

## Optional: speaker diarization

Diarization labels who is speaking, which makes transcript segments follow
speaker turns rather than Whisper's own chunking. It is **off by default
if no token is present**, and the pipeline continues without it.

To enable it:

1. Create a HuggingFace token: <https://huggingface.co/settings/tokens>
2. Accept the terms for both gated models (required, or loading 401s):
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
   - <https://huggingface.co/pyannote/segmentation-3.0>
3. Put the token in a `.env` file at the repo root:

   ```
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
   ```

`.env` is gitignored. Never commit a token.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ffmpeg not found on PATH` | Install ffmpeg (above) and reopen the terminal |
| `401` / `Could not download pyannote` | Token missing, or model terms not accepted — see above. Or just use `--no-diarize` |
| `ImportError: cannot import name 'VideoFileClip' from 'moviepy'` | moviepy 1.x installed; this needs 2.x. `pip install -U "moviepy>=2.0"` |
| `numpy.dtype size changed` | numpy 2.x ABI clash. `pip install "numpy<2"` |
| Download fails on a valid URL | `pip install -U yt-dlp` — extractors break when YouTube changes |
| Clips overlap heavily | Known limitation: matched windows are not merged yet |

## Known limitations

- Overlapping matches produce overlapping clips; intervals are not merged.
- Output keeps the source aspect ratio. No 9:16 reframing yet.
- Subtitles are written as sidecar `.srt`, not burned into the video.
- Models reload on every invocation — fine for a CLI, not for a service.
- Extracted `.wav` files are not cleaned up.

## Licence

MIT — see [LICENSE](LICENSE).
