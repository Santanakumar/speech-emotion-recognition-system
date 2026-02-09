import torch
import whisper
from transformers import pipeline
from pyannote.audio import Pipeline

from config import HUGGINGFACE_TOKEN


def load_models():
    """
    Loads Whisper, Emotion Classifier, and Speaker Diarization models.
    Returns all models as a tuple.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device set to: {device}")

    # Whisper Model
    print("Loading Whisper model...")
    whisper_model = whisper.load_model("large", device=device)

    # Emotion Classifier
    print("Loading Emotion Analyzer...")
    emotion_analyzer = pipeline(
        "text-classification",
        model="michellejieli/emotion_text_classifier",
        device=0 if torch.cuda.is_available() else -1,
    )

    # Speaker Diarization
    print("Loading Speaker Diarization Pipeline...")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HUGGINGFACE_TOKEN
    )
    diarization_pipeline.to(torch.device(device))

    print("All models loaded successfully!")

    return whisper_model, emotion_analyzer, diarization_pipeline
