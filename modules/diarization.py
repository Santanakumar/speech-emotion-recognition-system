from collections import defaultdict


def merge_speakers(diarization, max_speakers=2):
    """
    Merges speakers based on total speech duration.
    Maps dominant speakers into Speaker 1, Speaker 2...
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
