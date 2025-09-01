
from video_downloader import VideoDownloader
from transcriber import transcribe_video, save_transcription
from clip_finder import build_index, search_segments

vidDownloaded = True #if you want to use downloaded files for transcription
video_path = "downloads/Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster).mp4"

if __name__ == "__main__":
    if vidDownloaded:
        print(f"Using downloaded video at: {video_path}")
    else:
        downloader = VideoDownloader()
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # example
        video_path = downloader.download(url)
        print(f"Downloaded video saved at: {video_path}")

    # Transcribe the downloaded video
    print("Transcribing video...")
    transcription = transcribe_video(video_path)
    output_path = video_path + "_transcription.txt"
    save_transcription(transcription, output_path)
    print(f"Transcription saved at: {output_path}")

# Build semantic index
texts, embeddings = build_index(transcription)

# Example query
query = "love song"   # could be user input
results = search_segments(query, texts, embeddings, transcription)

# Print results
for r in results:
    print(f"[{r['start']:.2f} - {r['end']:.2f}] {r['text']} (score={r['score']:.2f})")
