import os
from datetime import timedelta

class ReelSubtitleAgent:

    def __init__(self, output_dir="subtitles"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def format_timestamp(self, seconds):
        """
        Convert seconds to SRT timestamp format: HH:MM:SS,mmm
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


    def save_srt(self, transcription_segments, clip_start, clip_end, output_name="clip_subtitles.srt"):
        """
        Generate subtitles for a specific clip based on start and end times.
        Adjusts timestamps to start from 00:00:00,000 for the clip.
        """
        output_path = os.path.join(self.output_dir, output_name)
        
        # Filter segments that fall within the clip timeframe
        clip_segments = []
        for segment in transcription_segments:
            seg_start = segment["start"]
            seg_end = segment["end"]
            
            # Check if segment overlaps with clip
            if seg_start < clip_end and seg_end > clip_start:
                # Adjust timestamps relative to clip start
                adjusted_start = max(0, seg_start - clip_start)
                adjusted_end = min(clip_end - clip_start, seg_end - clip_start)
                
                clip_segments.append({
                    "start": adjusted_start,
                    "end": adjusted_end,
                    "text": segment["text"]
                })
        print(clip_segments)
        
        # Write SRT file
        with open(output_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(clip_segments, 1):
                start = self.format_timestamp(segment["start"])
                end = self.format_timestamp(segment["end"])
                text = segment["text"]

                f.write(f"{i}\n")                    # Subtitle number on its own line
                f.write(f"{start} --> {end}\n")      # Timestamp on second line
                f.write(f"{text}\n\n")               # Text on third line, then blank line
        
        print(f"✅ Clip subtitles saved at {output_path}")
        return output_path

