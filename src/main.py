# src/main.py
from video_downloader import VideoDownloader
from transcriber import transcribe_video, save_transcription
from clip_finder import ClipFinder
from video_editor import VideoEditor
import os

vidDownloaded = False #if you want to use downloaded files for transcription
video_path = "downloads/cid.mp4"

if __name__ == "__main__":
    if vidDownloaded:
        print(f"Using downloaded video at: {video_path}")
    else:
        downloader = VideoDownloader()
        url = "https://www.youtube.com/watch?v=0dyEOsmX4iA"  # example
        video_path = downloader.download(url)
        print(f"Downloaded video saved at: {video_path}")

    # ==== Step 2: Transcribe video ====
    transcription_segments = transcribe_video(video_path, model_name="base")
    transcript_path = f"{video_path}_transcription.txt"
    save_transcription(transcription_segments, transcript_path)
    print(f"Transcription saved at: {transcript_path}")

    # ==== Step 3: Keyword search ====
    keyword = "flamethrower"  # you could replace with user input
    finder = ClipFinder()
    matched_segments = finder.search_segments(transcription_segments, keyword)
    print(f"Found {len(matched_segments)} segments for keyword '{keyword}'")

    # ==== Step 4: Clip extraction ====
    editor = VideoEditor()
    clips = editor.batch_cut_clips(
        video_path,
        [(seg["start"], seg["end"]) for seg in matched_segments],
        padding=2.0
    )
    print("Generated clips:", clips)