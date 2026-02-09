import os
from pydub import AudioSegment
from config import ALLOWED_EXTENSIONS


def allowed_file(filename):
    """Check if uploaded file format is supported."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def convert_audio_to_wav(file_path):
    """
    Converts input audio file into WAV format.
    Ensures compatibility with Whisper + PyAnnote.
    """
    base, ext = os.path.splitext(file_path)

    if ext.lower() == ".wav":
        return file_path

    wav_file = f"{base}.wav"

    try:
        audio = AudioSegment.from_file(file_path)
        audio.export(wav_file, format="wav")
        return wav_file
    except Exception as e:
        print(f"Audio conversion error: {e}")
        return None
