"""ViralClip-AI command line entry point.

Find moments in a long video that match a search query, and cut them into
short, subtitled clips.

    python src/main.py --url "https://www.youtube.com/watch?v=..." -q "flamethrower"
    python src/main.py --file downloads/podcast.mp4 -q "space travel" --dry-run
"""

import argparse
import sys
from pathlib import Path

from pipeline import PipelineError, Stage, run_pipeline
from video_editor import (
    DEFAULT_MAX_LEN,
    DEFAULT_MIN_LEN,
    DEFAULT_PADDING,
    ffmpeg_available,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="viralclip",
        description="Extract short, subtitled clips from a long video by search query.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python src/main.py --url https://youtu.be/0dyEOsmX4iA -q flamethrower\n"
            "  python src/main.py --file downloads/talk.mp4 -q \"climate policy\" --dry-run\n"
            "  python src/main.py --file downloads/talk.mp4 -q rockets --no-diarize -k 3\n"
        ),
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="Video URL to download (YouTube and others, via yt-dlp).")
    source.add_argument("--file", help="Path to a local video file already on disk.")

    parser.add_argument(
        "-q", "--query", required=True,
        help="What to look for in the transcript. Matched by keyword AND meaning.",
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=5, metavar="N",
        help="Maximum number of clips to produce (default: 5).",
    )
    parser.add_argument(
        "--model", default="base", choices=WHISPER_MODELS,
        help="Whisper model size. Bigger is more accurate and much slower (default: base).",
    )
    parser.add_argument(
        "--no-diarize", action="store_true",
        help="Skip speaker diarization. Avoids needing a HuggingFace token.",
    )
    parser.add_argument(
        "--padding", type=float, default=DEFAULT_PADDING, metavar="SEC",
        help=f"Seconds of context kept on each side of a match (default: {DEFAULT_PADDING}).",
    )
    parser.add_argument(
        "--min-len", type=float, default=DEFAULT_MIN_LEN, metavar="SEC",
        help=f"Minimum clip length (default: {DEFAULT_MIN_LEN}).",
    )
    parser.add_argument(
        "--max-len", type=float, default=DEFAULT_MAX_LEN, metavar="SEC",
        help=f"Maximum clip length (default: {DEFAULT_MAX_LEN}).",
    )
    parser.add_argument(
        "-o", "--output-dir", default=str(PROJECT_ROOT / "output"), metavar="DIR",
        help="Where downloads/, clips/ and subtitles/ are written (default: ./output).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Transcribe and search, print the clip windows, but cut nothing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.min_len > args.max_len:
        print(f"error: --min-len ({args.min_len}) exceeds --max-len ({args.max_len})",
              file=sys.stderr)
        return 2

    if not args.dry_run and not ffmpeg_available():
        print("error: ffmpeg not found on PATH. It is required to cut clips.\n"
              "       Install it (see README.md), or use --dry-run to preview windows.",
              file=sys.stderr)
        return 2

    # Pre-flight: catch a bad path as a usage error (exit 2) before the
    # pipeline loads any model weights. run_pipeline() checks this too, but
    # by then we have paid for a ~150 MB Whisper load.
    if args.file and not Path(args.file).is_file():
        print(f"error: no such file: {args.file}", file=sys.stderr)
        return 2

    out = Path(args.output_dir)
    for sub in ("downloads", "clips", "subtitles"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    def report(stage: Stage, percent: int, detail: str) -> None:
        print(f"[{percent:3d}%] {stage.value:<13} {detail}")

    try:
        result = run_pipeline(
            query=args.query,
            url=args.url,
            file=args.file,
            output_dir=out,
            top_k=args.top_k,
            whisper_model=args.model,
            diarize=not args.no_diarize,
            padding=args.padding,
            min_len=args.min_len,
            max_len=args.max_len,
            dry_run=args.dry_run,
            progress=report,
        )
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not result.clips:
        print(f"\nNo matches for {args.query!r}. Try a broader query.")
        return 0

    print()
    if args.dry_run:
        print("Windows that would be cut:")
        for clip in result.clips:
            print(f"  {clip.stem:<20} {clip.start:7.2f}s -> {clip.end:7.2f}s"
                  f"  score={clip.score:.3f}")
            print(f"  {'':20} {clip.text[:70]!r}")
        print("\nNothing written (dry run).")
        return 0

    print(f"Done. {len(result.clips)} clip(s):")
    for clip in result.clips:
        print(f"  {clip.clip_path}")
        print(f"  {clip.subtitle_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
