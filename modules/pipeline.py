from modules.preprocessing import convert_audio_to_wav
from modules.diarization import merge_speakers
from modules.emotion_analysis import analyze_emotions


def process_audio(file_path, whisper_model, diarization_pipeline, emotion_analyzer):
    """
    Full processing pipeline:
    Audio → WAV Conversion → Diarization → Transcription → Emotion Detection
    """

    wav_file_path = convert_audio_to_wav(file_path)
    if not wav_file_path:
        return None, None

    diarization = diarization_pipeline(wav_file_path)
    speaker_mapping = merge_speakers(diarization)

    result = whisper_model.transcribe(wav_file_path)

    segments, overall_emotion = analyze_emotions(
        result,
        diarization,
        speaker_mapping,
        emotion_analyzer
    )

    return segments, overall_emotion
