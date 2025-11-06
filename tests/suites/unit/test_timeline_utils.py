from ser.utils.timeline_utils import build_timeline


def test_build_timeline_merges_transcript_and_emotions():
    transcript = [("hello", 0.0, 0.5), ("world", 2.0, 2.5)]
    emotions = [("happy", 0.0, 1.0), ("sad", 3.0, 4.0)]

    timeline = build_timeline(transcript, emotions)

    assert timeline == [
        (0.0, "happy", "hello"),
        (1.0, "", ""),
        (2.0, "", "world"),
        (3.0, "sad", ""),
        (4.0, "", ""),
    ]
