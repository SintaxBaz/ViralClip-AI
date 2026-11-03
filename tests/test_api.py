"""HTTP API tests.

The pipeline is replaced with a fake, so the full request/response surface
is exercised without Whisper, torch or ffmpeg. Needs only fastapi + httpx.

    python tests/test_api.py
"""

import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# Stub the heavy imports pulled in transitively by pipeline.py.
for name in ("whisper", "pyannote", "pyannote.audio", "sentence_transformers",
             "moviepy", "dotenv", "yt_dlp", "torch"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["moviepy"].VideoFileClip = object
sys.modules["dotenv"].load_dotenv = lambda *a, **k: None
sys.modules["sentence_transformers"].SentenceTransformer = object
sys.modules["sentence_transformers"].util = types.SimpleNamespace(semantic_search=None)
sys.modules["pyannote.audio"].Pipeline = object

import httpx                                                     # noqa: E402
from fastapi.testclient import TestClient                        # noqa: E402

from api.app import create_app                                   # noqa: E402
from api.config import Settings                                  # noqa: E402
from pipeline import ClipResult, PipelineError, PipelineResult, Stage  # noqa: E402

passed = failed = 0


def check(label, got, want):
    global passed, failed
    if got == want:
        passed += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}\n          got:  {got!r}\n          want: {want!r}")


TMP = Path("/tmp/vc_api_test")


def fake_pipeline(*, query, output_dir, progress=None, **kw):
    """Stand-in for run_pipeline: reports progress, writes two real files."""
    if query == "boom":
        raise PipelineError("Simulated pipeline failure")
    if query == "crash":
        raise ValueError("unexpected internal explosion")

    out = Path(output_dir)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "subtitles").mkdir(parents=True, exist_ok=True)

    if progress:
        progress(Stage.TRANSCRIBING, 30, "faking transcription")
        progress(Stage.CUTTING, 70, "faking cuts")

    if query == "nomatch":
        return PipelineResult(source_path="src.mp4", transcript_path="t.txt",
                              segment_count=3)

    (out / "clips" / "clip_1_16.mp4").write_bytes(b"\x00fake mp4 payload")
    (out / "subtitles" / "clip_1_16.srt").write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nhello\n\n", encoding="utf-8")

    return PipelineResult(
        source_path="src.mp4", transcript_path="t.txt", segment_count=3,
        clips=[ClipResult(stem="clip_1_16", start=1.0, end=16.0, score=0.9,
                          text="a matched line",
                          clip_path=str(out / "clips" / "clip_1_16.mp4"),
                          subtitle_path=str(out / "subtitles" / "clip_1_16.srt"))],
    )


settings = Settings(output_root=TMP, warm_models=False, max_upload_bytes=2048)
client = TestClient(create_app(settings=settings, pipeline_fn=fake_pipeline, warm=False))


def wait_for(job_id, timeout=5.0):
    """Poll until terminal -- the same loop a real client would run."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never finished: {body}")


with client:
    print("healthz")
    r = client.get("/healthz")
    check("returns a status", r.json()["status"] in ("ok", "degraded"), True)
    check("reports concurrency", r.json()["concurrency"], 1)

    print("\nPOST /jobs  (url submission)")
    r = client.post("/jobs", json={"url": "https://e.com/v", "query": "rockets"})
    check("202 Accepted", r.status_code, 202)
    job = r.json()
    check("returns a job id", bool(job["id"]), True)
    check("starts non-terminal", job["status"] in ("queued", "downloading"), True)
    check("echoes the query", job["query"], "rockets")
    check("records the source", job["source"],
          {"kind": "url", "ref": "https://e.com/v"})

    print("\nthe job runs to completion")
    done = wait_for(job["id"])
    check("reaches done", done["status"], "done")
    check("percent is 100", done["percent"], 100)
    check("has a finished_at", done["finished_at"] is not None, True)
    check("result has one clip", len(done["result"]["clips"]), 1)

    print("\nGET /jobs/{id}/clips")
    body = client.get(f"/jobs/{job['id']}/clips").json()
    check("lists the clip", body["clips"][0]["clip"], "clip_1_16.mp4")
    check("pairs the subtitle", body["clips"][0]["subtitle"], "clip_1_16.srt")
    check("gives a download url",
          body["clips"][0]["download"].endswith(f"/jobs/{job['id']}/clips/clip_1_16.mp4"),
          True)
    check("reports duration", body["clips"][0]["duration"], 15.0)

    print("\nGET /jobs/{id}/clips/{name}  (download)")
    r = client.get(f"/jobs/{job['id']}/clips/clip_1_16.mp4")
    check("200 for the clip", r.status_code, 200)
    check("mp4 content type", r.headers["content-type"], "video/mp4")
    check("body is the file", r.content, b"\x00fake mp4 payload")
    r = client.get(f"/jobs/{job['id']}/clips/clip_1_16.srt")
    check("200 for the srt", r.status_code, 200)
    check("srt content type", r.headers["content-type"], "application/x-subrip")

    print("\npath traversal is refused")
    for bad in ("../../../etc/passwd", "..%2F..%2Fetc%2Fpasswd", "a/b.mp4"):
        r = client.get(f"/jobs/{job['id']}/clips/{bad}")
        check(f"rejects {bad!r}", r.status_code in (400, 404), True)

    print("\nupload submission")
    r = client.post("/jobs/upload",
                    files={"file": ("my clip!.mp4", b"\x00" * 512, "video/mp4")},
                    data={"query": "rockets"})
    check("202 Accepted", r.status_code, 202)
    up = r.json()
    check("source kind is upload", up["source"]["kind"], "upload")
    check("filename sanitized", "my clip!" not in up["source"]["ref"], True)
    check("upload job completes", wait_for(up["id"])["status"], "done")

    print("\nupload validation")
    r = client.post("/jobs/upload",
                    files={"file": ("notes.txt", b"hello", "text/plain")},
                    data={"query": "x"})
    check("415 on bad extension", r.status_code, 415)
    r = client.post("/jobs/upload",
                    files={"file": ("big.mp4", b"\x00" * 4096, "video/mp4")},
                    data={"query": "x"})
    check("413 over the size cap", r.status_code, 413)
    r = client.post("/jobs/upload",
                    files={"file": ("empty.mp4", b"", "video/mp4")},
                    data={"query": "x"})
    check("422 on empty file", r.status_code, 422)
    r = client.post("/jobs/upload",
                    files={"file": ("ok.mp4", b"\x00" * 16, "video/mp4")}, data={})
    check("422 with no query", r.status_code, 422)

    print("\nrequest validation")
    check("422 missing url and query", client.post("/jobs", json={}).status_code, 422)
    check("422 non-http url",
          client.post("/jobs", json={"url": "file:///etc/passwd", "query": "a"}).status_code,
          422)
    check("422 top_k over the cap",
          client.post("/jobs", json={"url": "https://e.com/v", "query": "a",
                                     "top_k": 99}).status_code, 422)
    check("422 min_len above max_len",
          client.post("/jobs", json={"url": "https://e.com/v", "query": "a",
                                     "min_len": 40, "max_len": 10}).status_code, 422)
    check("404 unknown job", client.get("/jobs/deadbeef").status_code, 404)
    check("404 clips of unknown job",
          client.get("/jobs/deadbeef/clips").status_code, 404)

    print("\nfailures surface as a failed job, not a 500")
    r = client.post("/jobs", json={"url": "https://e.com/v", "query": "boom"})
    check("still 202", r.status_code, 202)
    bad = wait_for(r.json()["id"])
    check("status failed", bad["status"], "failed")
    check("error is the pipeline message", bad["error"], "Simulated pipeline failure")
    check("409 when listing its clips",
          client.get(f"/jobs/{r.json()['id']}/clips").status_code, 409)

    print("\nunexpected errors do not leak internals")
    r = client.post("/jobs", json={"url": "https://e.com/v", "query": "crash"})
    crashed = wait_for(r.json()["id"])
    check("status failed", crashed["status"], "failed")
    check("error is redacted", crashed["error"], "Internal error: ValueError")
    check("no traceback leaked", "explosion" in str(crashed), False)

    print("\nzero matches is success, not failure")
    r = client.post("/jobs", json={"url": "https://e.com/v", "query": "nomatch"})
    empty = wait_for(r.json()["id"])
    check("status done", empty["status"], "done")
    check("no clips", empty["result"]["clips"], [])

    print("\nGET /jobs")
    jobs = client.get("/jobs").json()["jobs"]
    # Exactly the five that returned 202: url, upload, boom, crash, nomatch.
    # Every rejected submission (415/413/422) must leave no job behind.
    check("lists only accepted jobs", len(jobs), 5)
    check("newest first",
          jobs[0]["created_at"] >= jobs[-1]["created_at"], True)
    check("honours limit", len(client.get("/jobs?limit=2").json()["jobs"]), 2)

    print("\njob isolation")
    a = client.post("/jobs", json={"url": "https://e.com/a", "query": "rockets"}).json()
    b = client.post("/jobs", json={"url": "https://e.com/b", "query": "rockets"}).json()
    wait_for(a["id"]); wait_for(b["id"])
    check("each job gets its own directory", a["id"] != b["id"], True)
    check("job A cannot serve job B's file",
          client.get(f"/jobs/{a['id']}/clips/clip_1_16.mp4").status_code, 200)
    check("both directories exist",
          (TMP / a["id"] / "clips" / "clip_1_16.mp4").is_file()
          and (TMP / b["id"] / "clips" / "clip_1_16.mp4").is_file(), True)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
