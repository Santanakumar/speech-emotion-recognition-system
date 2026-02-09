import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from collections import Counter

from config import UPLOAD_FOLDER
from models import load_models
from utils import allowed_file, convert_audio_to_wav, merge_speakers

# Flask App
app = Flask(__name__)

# Load Models
whisper_model, emotion_analyzer, diarization_pipeline = load_models()


# Home Page
@app.route("/")
def index():
    return render_template("index.html")


# Upload Route
@app.route("/upload", methods=["POST"])
def upload_file():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported format"}), 400

    # Save File
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Convert Audio
    wav_file_path = convert_audio_to_wav(file_path)
    if not wav_file_path:
        return jsonify({"error": "Audio conversion failed"}), 500

    # Speaker Diarization
    diarization = diarization_pipeline(wav_file_path)
    speaker_mapping = merge_speakers(diarization)

    # Transcription
    result = whisper_model.transcribe(wav_file_path)

    segments = []
    emotion_counts = Counter()

    # Emotion Analysis
    for segment in result.get("segments", []):

        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]

        best_speaker, max_overlap = None, 0

        for turn, _, speaker in diarization.itertracks(yield_label=True):

            overlap_start = max(turn.start, start_time)
            overlap_end = min(turn.end, end_time)

            overlap_duration = max(0, overlap_end - overlap_start)

            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_speaker = speaker

        speaker_label = speaker_mapping.get(best_speaker, "Unknown")

        emotions = emotion_analyzer(text)
        detected_emotion = emotions[0]["label"]

        emotion_counts[detected_emotion] += 1

        segments.append({
            "speaker": speaker_label,
            "start": start_time,
            "end": end_time,
            "text": text,
            "emotion": detected_emotion
        })

    overall_emotion = emotion_counts.most_common(1)[0][0]

    return jsonify({
        "segments": segments,
        "overall_emotion": overall_emotion
    })


if __name__ == "__main__":
    app.run(debug=True)
