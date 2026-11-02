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
| `src/pipeline.py` | The pipeline itself, shared by the CLI and the API |
| `src/main.py` | CLI entry point |
| `src/api/app.py` | FastAPI routes and app wiring |
| `src/api/jobs.py` | Job records and the job store |
| `src/api/worker.py` | Bounded worker pool |
| `src/api/config.py` | Settings from the environment |
| `Dockerfile` | Multi-stage build; models baked in, CPU torch |
| `docker/preload_models.py` | Build-time weight download |
| `.github/workflows/ci.yml` | Tests, Docker build, container smoke test |

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
```

Activate it. **Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

Use the `.ps1`, not bare `.venv\Scripts\activate` — that resolves to
`activate.bat`, which PowerShell runs in a child `cmd` process that then
exits, leaving the venv *not* activated. If PowerShell refuses with
"running scripts is disabled on this system", allow it for this window only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**Windows cmd.exe:** `.venv\Scripts\activate.bat`
**macOS / Linux:** `source .venv/bin/activate`

Then:

```bash
pip install -r requirements.txt
```

Speaker diarization is an optional extra and is **not** installed by the
line above, because `pyannote.audio` pulls a large dependency tree and
needs a HuggingFace token. Everything works without it. To add it:

```bash
pip install -r requirements-diarize.txt
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

## HTTP API

The CLI and the API share one pipeline implementation (`src/pipeline.py`),
so they cannot drift.

```bash
uvicorn api.app:app --app-dir src --reload
```

Interactive docs at <http://127.0.0.1:8000/docs>.

### Why it is a job queue and not just an endpoint

A 40-minute video takes minutes to transcribe. Two constraints follow:

1. **It cannot run inside the request.** The client, and every proxy
   between it and the server, times out long before the work is done. So
   `POST /jobs` validates, enqueues, and returns `202` with a job id. The
   client polls.
2. **It cannot run on the event loop.** `async def` does not make blocking
   work concurrent. A synchronous Whisper call in a coroutine freezes the
   loop and stalls every other request, including the status polls. So the
   pipeline runs in a thread pool and the loop stays free.

Concurrency defaults to **1**. Model inference keeps its weights resident
for the whole run, so raising it multiplies peak memory rather than
throughput — scale with replicas, not threads. Models load once at
startup, not per request.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Readiness; `503` if ffmpeg is missing |
| `POST` | `/jobs` | Submit from a URL; `202` + job id |
| `POST` | `/jobs/upload` | Submit a `multipart/form-data` upload |
| `GET` | `/jobs` | Recent jobs, newest first |
| `GET` | `/jobs/{id}` | Status, percent, result |
| `GET` | `/jobs/{id}/clips` | Clip list with download URLs |
| `GET` | `/jobs/{id}/clips/{name}` | Download one clip or `.srt` |

### Example

```bash
# Submit a URL
curl -s -X POST http://127.0.0.1:8000/jobs \
  -H 'content-type: application/json' \
  -d '{"url":"https://www.youtube.com/watch?v=0dyEOsmX4iA","query":"flamethrower"}'
# -> {"id":"a1b2c3d4e5f6","status":"queued","percent":0, ...}

# Or upload a file
curl -s -X POST http://127.0.0.1:8000/jobs/upload \
  -F file=@podcast.mp4 -F query=rockets -F top_k=3

# Poll it
curl -s http://127.0.0.1:8000/jobs/a1b2c3d4e5f6
# -> {"status":"transcribing","percent":30,"detail":"..."}

# Fetch the clips once done
curl -s http://127.0.0.1:8000/jobs/a1b2c3d4e5f6/clips
curl -sO http://127.0.0.1:8000/jobs/a1b2c3d4e5f6/clips/clip_11_26.mp4
```

Job status is one of `queued`, `downloading`, `transcribing`, `searching`,
`cutting`, `subtitling`, `done`, `failed`. A pipeline failure becomes a
`failed` job with a message — not a `500`.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `VIRALCLIP_OUTPUT_ROOT` | `./output/jobs` | Where job output is written |
| `VIRALCLIP_CONCURRENCY` | `1` | Videos processed at once |
| `VIRALCLIP_MAX_QUEUE` | `32` | Queue depth before `503` |
| `VIRALCLIP_MAX_UPLOAD_BYTES` | `2147483648` | Upload cap (2 GiB) |
| `VIRALCLIP_WHISPER_MODEL` | `base` | Model size used by the service |
| `VIRALCLIP_DIARIZE` | `false` | Enable diarization (needs a token) |
| `VIRALCLIP_WARM_MODELS` | `true` | Load models at startup |

### Not yet production

Jobs live in process memory, so a restart loses them and a second replica
cannot see them. `JobStore` has five methods and the API never touches its
dict directly, so swapping in Redis or Postgres is contained. There is also
no authentication, no rate limiting and no cleanup of old job output.

## CI

`.github/workflows/ci.yml` runs on every push and PR:

1. **Unit + API tests** — the three fast suites, with only `fastapi` and
   `httpx` installed (torch, whisper and pyannote are stubbed, so there is
   no reason to pay for them here).
2. **Docker build and smoke test** — builds the real image with
   `WHISPER_MODEL=tiny` (fast, same multi-stage path as production), starts
   the container, waits for `/healthz` to report healthy, uploads a
   synthetic clip generated on the runner, and polls the job to `done`
   through the real HTTP API.

Step 2 is the one a machine without Docker installed cannot run locally —
it is the actual proof the Dockerfile works, on infrastructure neither of
us controls.

## Docker

```bash
docker build -t viralclip .
docker run --rm -p 8000:8000 -v viralclip-data:/data viralclip
```

Or with compose:

```bash
docker compose up --build
```

Then <http://127.0.0.1:8000/docs>. No Python, no ffmpeg and no HuggingFace
account needed on the host.

### What the image does and why

**ffmpeg is installed in the runtime stage, not just the builder.** It is a
system binary needed at run time to cut clips, and `/healthz` returns `503`
without it.

**torch comes from PyTorch's CPU index.** The default Linux torch wheel is a
CUDA build of roughly 2.5 GB; the CPU wheel is around 200 MB. For a GPU
host, build with `--build-arg TORCH_INDEX=` (empty) to get the CUDA wheel
back.

**Model weights are downloaded at build time**, so they live in an image
layer. Otherwise the first request to every replica pays for a ~240 MB
download, and the image cannot start at all on a host without egress to
HuggingFace.

**Requirements are copied and installed before the source**, so editing a
`.py` file does not reinstall torch — the difference between a 2-second and
a 10-minute rebuild.

**It is a multi-stage build.** Compilers and `git` stay in the builder; the
runtime stage gets the finished venv and model cache only.

**It runs as a non-root user** (uid 10001), with `/data` and the model cache
owned by that user so job output and a differently configured model still
work.

**The server binds `0.0.0.0`.** Bound to `127.0.0.1`, a container's server
is unreachable from the host even with `-p` published.

**Diarization is excluded by default** — it roughly doubles the image and
needs a token. Build with `--build-arg WITH_DIARIZATION=true` to include it.

### Build arguments

| Argument | Default | Meaning |
|---|---|---|
| `PYTHON_VERSION` | `3.11` | Base image tag |
| `WHISPER_MODEL` | `base` | Model baked into the image |
| `WITH_DIARIZATION` | `false` | Install `pyannote.audio` |
| `TORCH_INDEX` | PyTorch CPU index | Set empty for the CUDA wheel |
| `TORCH_VERSION` | `>=2.0.0,<3` | torch constraint |

### Volumes

Job output is written to `/data/jobs`, mounted as the `viralclip-data`
volume, so finished clips survive a restart. The job *registry* does not —
it is in process memory, so in-flight jobs are lost on restart.

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
| `running scripts is disabled on this system` | PowerShell execution policy. `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `The token '&&' is not a valid statement separator` | Windows PowerShell 5.1 has no `&&`. Run the commands on separate lines |
| venv seems active but packages are missing | You ran `activate.bat` from PowerShell. Use `.\.venv\Scripts\Activate.ps1` |
| Download fails on a valid URL | `pip install -U yt-dlp` — extractors break when YouTube changes |
| Clips overlap heavily | Known limitation: matched windows are not merged yet |

## Tests

Both suites stub the heavy imports, so they run in about a second with no
model downloads and no ffmpeg:

```bash
python tests/test_smoke.py          # clip maths, naming, SRT, CLI validation
python tests/test_api.py            # full HTTP surface against a fake pipeline
python tests/test_optional_deps.py  # diarization really is optional
python tests/test_ffmpeg_integration.py  # real ffmpeg: generates a video, cuts it, checks durations
```

`test_ffmpeg_integration.py` needs ffmpeg and skips cleanly without it.

Or inside the image:

```bash
docker run --rm viralclip python tests/test_api.py
```

## Known limitations

- Overlapping matches produce overlapping clips; intervals are not merged.
- Output keeps the source aspect ratio. No 9:16 reframing yet.
- Subtitles are written as sidecar `.srt`, not burned into the video.
- Clip selection is query-driven; there is no automatic highlight ranking.
- API jobs are held in memory (see above).
- Extracted `.wav` files and finished job directories are not cleaned up.

## Licence

MIT — see [LICENSE](LICENSE).
