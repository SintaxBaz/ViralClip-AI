
from video_downloader import VideoDownloader
from transcriber import transcribe_video, save_transcription

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

    # Transcribe the downloaded video
    print("Transcribing video...")
    transcription = transcribe_video(video_path)
    output_path = video_path + "_transcription.txt"
    save_transcription(transcription, output_path)
    print(f"Transcription saved at: {output_path}")
