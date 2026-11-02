"""ViralClip-AI command line entry point.

Find moments in a long video that match a search query, and cut them into
short, subtitled clips.

    python src/main.py --url "https://www.youtube.com/watch?v=..." -q "flamethrower"
    python src/main.py --file downloads/podcast.mp4 -q "space travel" --dry-run
"""

import argparse
import sys
from pathlib import Path

from clip_finder import ClipFinder
from subtitle_agent import ReelSubtitleAgent
from transcriber import save_transcription, transcribe_video
from video_downloader import VideoDownloader
from video_editor import (
    DEFAULT_MAX_LEN,
    DEFAULT_MIN_LEN,
    DEFAULT_PADDING,
    VideoEditor,
    clip_stem,
    ffmpeg_available,
    resolve_window,
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

    out = Path(args.output_dir)
    for sub in ("downloads", "clips", "subtitles"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # ---- Step 1: obtain the source video ----
    if args.file:
        video_path = Path(args.file)
        if not video_path.is_file():
            print(f"error: no such file: {video_path}", file=sys.stderr)
            return 2
        video_path = str(video_path)
        print(f"[1/5] Using local file: {video_path}")
    else:
        print(f"[1/5] Downloading {args.url}")
        video_path = VideoDownloader(download_path=str(out / "downloads")).download(args.url)
        print(f"      Saved to {video_path}")

    # ---- Step 2: transcribe ----
    print(f"[2/5] Transcribing with Whisper '{args.model}'"
          f"{'' if args.no_diarize else ' + speaker diarization'} ...")
    segments = transcribe_video(video_path, model_name=args.model,
                                diarize=not args.no_diarize)
    if not segments:
        print("error: transcription produced no segments.", file=sys.stderr)
        return 1

    transcript_path = out / f"{Path(video_path).stem}_transcription.txt"
    save_transcription(segments, str(transcript_path))
    print(f"      {len(segments)} segments -> {transcript_path}")

    # ---- Step 3: search ----
    print(f"[3/5] Searching for: {args.query!r}")
    matches = ClipFinder().search_segments(segments, args.query, top_k=args.top_k)
    if not matches:
        print(f"      No matches for {args.query!r}. Try a broader query.")
        return 0
    print(f"      {len(matches)} match(es)")

    # ---- Step 4: resolve windows (shared by clips and subtitles) ----
    windows = [
        resolve_window(m["start"], m["end"], args.padding, args.min_len, args.max_len)
        for m in matches
    ]

    if args.dry_run:
        print("[4/5] Dry run -- windows that would be cut:")
        for (start, end), m in zip(windows, matches):
            print(f"      {clip_stem(start, end):<22} "
                  f"{start:7.2f}s -> {end:7.2f}s  score={m['score']:.3f}")
            print(f"      {'':22} {m['text'][:70]!r}")
        print("[5/5] Nothing written (dry run).")
        return 0

    # ---- Step 4b: cut clips ----
    print(f"[4/5] Cutting {len(windows)} clip(s) ...")
    editor = VideoEditor(output_dir=str(out / "clips"))
    clips = editor.batch_cut_clips(video_path, windows)
    for path in clips:
        print(f"      {path}")

    # ---- Step 5: subtitles, named to match their clip ----
    print(f"[5/5] Writing subtitles ...")
    agent = ReelSubtitleAgent(output_dir=str(out / "subtitles"))
    for start, end in windows:
        agent.save_srt(segments, start, end, output_name=f"{clip_stem(start, end)}.srt")

    print(f"\nDone. {len(clips)} clip(s) in {out / 'clips'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
