import whisper
from moviepy import VideoFileClip
import os

def extract_audio(video_path, audio_path):
	"""
	Extracts audio from a video file and saves it as a .wav file.
	"""
	clip = VideoFileClip(video_path)
	clip.audio.write_audiofile(audio_path)


def transcribe_video(video_path, model_name="base"):
	"""
	Transcribes the audio from a video file using Whisper.
	Returns the transcription as a list of segments with timestamps.
	"""
	audio_path = video_path + ".wav"
	extract_audio(video_path, audio_path)
	model = whisper.load_model(model_name)
	result = model.transcribe(audio_path, word_timestamps=True)
	os.remove(audio_path)  # Clean up the temporary audio file
	return result["segments"]


def save_transcription(transcription_segments, output_path):
	"""
	Saves the transcription with timestamps to a text file.
	"""
	with open(output_path, "w", encoding="utf-8") as f:
		for segment in transcription_segments:
			start = segment["start"]
			end = segment["end"]
			text = segment["text"]
			f.write(f"[{start:.2f} - {end:.2f}] {text}\n")
