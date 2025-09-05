# src/main.py
from video_downloader import VideoDownloader
from transcriber import transcribe_video, save_transcription
from clip_finder import ClipFinder
from video_editor import VideoEditor
from subtitle_agent import ReelSubtitleAgent
import os

padding = 2.0
min_len = 15.0
max_len = 25.0
# ==== Step 1: Download video ====
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
        padding= padding
    )
    print("Generated clips:", clips)

    # ==== Step 5: Subtitle Agent ====

    subtitle_agent = ReelSubtitleAgent(output_dir="subtitles")
    x = 1
    for clip in matched_segments:
        start = max(0,clip["start"]-padding)
        end = clip["end"] + padding

        duration = end - start
        if duration < min_len:
            end = start + min_len
        elif duration > max_len:
            end = start + max_len

        srt_path = subtitle_agent.save_srt(transcription_segments, start, end, output_name=f"sub_{round(start, 0)}_{round(end, 0)}.srt")
        x += 1

    print(matched_segments)
