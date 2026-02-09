import os
from collections import defaultdict
from pydub import AudioSegment

from config import ALLOWED_EXTENSIONS


def allowed_file(filename):
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_audio_to_wav(file_path):
    """
    Converts uploaded audio into WAV format (16kHz mono).
    """

    base, ext = os.path.splitext(file_path)

    if ext.lower() == ".wav":
        return file_path

    wav_file = f"{base}.wav"

    try:
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(wav_file, format="wav")
        print("Audio converted successfully.")
        return wav_file

    except Exception as e:
        print(f"Audio conversion error: {e}")
        return None


def merge_speakers(diarization, max_speakers=2):
    """
    Maps diarization speakers into Speaker 1, Speaker 2...
    based on speaking duration.
    """

    speaker_durations = defaultdict(float)

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_durations[speaker] += turn.end - turn.start

    sorted_speakers = sorted(
        speaker_durations,
        key=speaker_durations.get,
        reverse=True
    )[:max_speakers]

    return {
        speaker: f"Speaker {i+1}"
        for i, speaker in enumerate(sorted_speakers)
    }
