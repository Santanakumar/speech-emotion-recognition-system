from collections import Counter


def analyze_emotions(result, diarization, speaker_mapping, emotion_analyzer):
    """
    Assign emotions to each transcription segment
    and map them to the correct speaker.
    """

    segments = []
    emotion_counts = Counter()

    for segment in result.get("segments", []):
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]

        best_speaker = None
        max_overlap = 0

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            overlap_start = max(turn.start, start_time)
            overlap_end = min(turn.end, end_time)

            overlap_duration = max(0, overlap_end - overlap_start)

            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_speaker = speaker

        speaker_label = speaker_mapping.get(best_speaker, "Unknown")

        emotions = emotion_analyzer(text)
        detected_emotion = emotions[0]["label"] if emotions else "Neutral"

        emotion_counts[detected_emotion] += 1

        segments.append({
            "speaker": speaker_label,
            "start": start_time,
            "end": end_time,
            "text": text,
            "emotion": detected_emotion
        })

    overall_emotion = (
        emotion_counts.most_common(1)[0][0]
        if emotion_counts else "Neutral"
    )

    return segments, overall_emotion
