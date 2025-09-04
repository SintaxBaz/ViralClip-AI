# src/video_editor.py
import os
import subprocess
from typing import List, Tuple

class VideoEditor:
    def __init__(self, output_dir="clips"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def cut_clip(self, video_path: str, start: float, end: float, padding: float = 2.0, min_len: float = 15.0, max_len: float = 25.0) -> str:
        #""" Cuts a video segment ensuring clips are between 15–25 seconds long. """
        start_time = max(0, start - padding)
        end_time = end + padding
        duration = end_time - start_time

        # Enforce min/max clip length
        if duration < min_len:
            end_time = start_time + min_len
            duration = min_len
        elif duration > max_len:
            end_time = start_time + max_len
            duration = max_len

        filename = os.path.join(
            self.output_dir,
            f"clip_{int(start_time)}_{int(end_time)}.mp4"
        )

        command = [
            "ffmpeg", "-y", "-i", video_path,
            "-ss", str(start_time), "-t", str(duration),
            "-c:v", "libx264", "-c:a", "aac", filename
        ]
        subprocess.run(command, check=True)
        return filename


    def batch_cut_clips(self, video_path: str, segments: List[Tuple[float, float]], padding: float = 2.0) -> List[str]:
        """
        Cuts multiple clips from the video.
        
        Args:
            video_path (str): Path to the video file.
            segments (list of (start, end)): List of time ranges.
            padding (float): Seconds before and after each clip.
        
        Returns:
            list[str]: Paths to the generated clips.
        """
        clips = []
        for start, end in segments:
            clip_path = self.cut_clip(video_path, start, end, padding)
            clips.append(clip_path)
        return clips
