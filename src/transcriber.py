
import whisper
from moviepy import VideoFileClip
from pyannote.audio import Pipeline
from dotenv import load_dotenv
import os

def sanitize_filename(path):
    directory, filename = os.path.split(path)
    safe_filename = filename.replace(" ", "_")
    safe_path = os.path.join(directory, safe_filename)
    if not os.path.exists(safe_path):
        os.rename(path, safe_path)
    return safe_path


def extract_audio(video_path, audio_path):
	"""
	Extracts audio from a video file and saves it as a .wav file.
	"""
	video_path = sanitize_filename(video_path)
	clip = VideoFileClip(video_path)
	clip.audio.write_audiofile(audio_path)


def transcribe_video(video_path, model_name="base"):
	"""
	Transcribes the audio from a video file using Whisper.
	Returns the transcription as a list of segments with timestamps.
	Also runs speaker diarization and saves RTTM file.
	"""
	# Load environment variables from the root .env directory
	load_dotenv(dotenv_path="../.env/HF_TOKEN.ini")
	hf_token = os.getenv("HF_TOKEN")
	
	audio_path = video_path + ".wav"
	extract_audio(video_path, audio_path)
	model = whisper.load_model(model_name)
	result = model.transcribe(audio_path, word_timestamps=True)

	pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
	diarization = pipeline(audio_path)

	segments = result["segments"]

	final_segment = []

	for turn, _, speaker in diarization.itertracks(yield_label=True):
		start_time, end_time = turn.start, turn.end
		words_in_segment = []

		for segment in segments:  
			if "words" in segment:   # Whisper stores per-word timings here
				for word in segment["words"]:
					if not (word["end"] < start_time or word["start"] > end_time):
						words_in_segment.append(word["word"].strip())
		

		if words_in_segment:
			text = " ".join(words_in_segment)
			final_segment.append({
				"speaker": speaker,
				"start": start_time,
				"end": end_time,
            	"text": text
		})

	return final_segment


def save_transcription(transcription_segments, output_path):
	"""
	Saves the transcription with timestamps to a text file.
	"""
	with open(output_path, "w", encoding="utf-8") as f:
		for segment in transcription_segments:
			start = segment["start"]
			end = segment["end"]
			text = segment["text"]
			speaker = segment["speaker"]
			f.write(f"[{start:.2f} - {end:.2f}] {speaker}: {text}\n")
