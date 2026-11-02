"""Download model weights at build time so they live in an image layer.

Without this the container downloads ~240 MB on first request: the first
caller waits for it, it happens again in every replica, and the image
cannot run in a network-restricted environment at all.

Run during the build with XDG_CACHE_HOME / HF_HOME pointed at a directory
that is then copied into the runtime stage.
"""

import os
import sys


def main() -> int:
    whisper_model = os.environ.get("WHISPER_MODEL", "base")
    embedder = os.environ.get("EMBEDDER_MODEL", "all-MiniLM-L6-v2")

    print(f"Preloading Whisper '{whisper_model}' ...", flush=True)
    import whisper

    whisper.load_model(whisper_model)

    print(f"Preloading embedder '{embedder}' ...", flush=True)
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(embedder)

    cache = os.environ.get("XDG_CACHE_HOME", "(default)")
    print(f"Models cached under {cache}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
