# src/video_downloader.py
import os
import yt_dlp

class VideoDownloader:
    def __init__(self, download_path="downloads"):
        self.download_path = download_path
        os.makedirs(download_path, exist_ok=True)

    def download(self, url: str) -> str:
        """
        Download a video from a given URL (supports multiple platforms).
        Returns the path of the downloaded video file.
        """
        try:
            ydl_opts = {
                "outtmpl": os.path.join(self.download_path, "%(title)s.%(ext)s"),
                "format": "mp4/best",
                "quiet": True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                file_path = ydl.prepare_filename(info_dict)
                return file_path
        except Exception as e:
            raise RuntimeError(f"Failed to download video: {str(e)}")
