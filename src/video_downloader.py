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
                # "mp4/best" picks the single best stream already in mp4.
                # On many YouTube videos the highest-quality video is DASH
                # (video-only mp4 + separate audio-only m4a), so that
                # selector silently grabs a video with NO audio track --
                # the download succeeds and plays fine, it just has
                # nothing for Whisper to transcribe.
                #
                # Ask for the best video and best audio as two streams and
                # merge them (ffmpeg does the merge; we already require it
                # downstream to cut clips). Falls back to a single best
                # stream for a source that has no split streams at all.
                "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
                "merge_output_format": "mp4",
                "quiet": True,
                "no_warnings": True,
                # Emit ASCII filenames with no spaces, so downstream
                # ffmpeg calls and path handling stay simple.
                "restrictfilenames": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                file_path = ydl.prepare_filename(info_dict)
                # merge_output_format rewrites the extension to .mp4 after
                # prepare_filename() has already been computed from the
                # pre-merge info, so the two can disagree (e.g. a webm
                # container becomes .mp4 post-merge). Trust what is
                # actually on disk.
                merged = os.path.splitext(file_path)[0] + ".mp4"
                if merged != file_path and os.path.exists(merged):
                    return merged
                return file_path
        except Exception as e:
            raise RuntimeError(f"Failed to download video: {str(e)}")
