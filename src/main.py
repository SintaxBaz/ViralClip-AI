from video_downloader import VideoDownloader

if __name__ == "__main__":
    downloader = VideoDownloader()
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # example
    video_path = downloader.download(url)
    print(f"Downloaded video saved at: {video_path}")
