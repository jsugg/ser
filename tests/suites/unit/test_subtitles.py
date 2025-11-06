from pathlib import Path

import pytest

from ser.utils.subtitles import (
    ASSFormatter,
    SRTFormatter,
    VTTFormatter,
    parse_subtitles,
    timeline_to_subtitles,
)


def test_timeline_to_subtitles_sorts_and_computes_durations():
    timeline = [
        (5.0, "happy", "Third"),
        (1.0, "neutral", "First"),
        (3.0, "sad", "Second"),
    ]

    result = timeline_to_subtitles(timeline, default_duration=2.0)

    assert [start for start, *_ in result] == [1.0, 3.0, 5.0]
    assert result[0][1] == pytest.approx(2.0)
    assert result[1][1] == pytest.approx(2.0)


def test_timeline_to_subtitles_uses_default_duration_for_last_entry():
    timeline = [(0.0, "calm", "Only line")]

    result = timeline_to_subtitles(timeline, default_duration=3.5)

    assert result == [(0.0, 3.5, "Only line", "calm")]


def test_timeline_to_subtitles_skips_blank_text():
    timeline = [
        (0.0, "happy", "  Hello "),
        (1.0, "sad", "   "),
        (2.0, "angry", "World"),
    ]

    result = timeline_to_subtitles(timeline, default_duration=1.0)

    assert result == [
        (0.0, 1.0, "Hello", "happy"),
        (2.0, 1.0, "World", "angry"),
    ]


def test_parse_subtitles_handles_invalid_entries(caplog):
    caplog.set_level("ERROR")
    parsed = parse_subtitles("0.0,1.0,Hello,Happy;invalid;2.0,1.5,Bye,Sad")

    assert parsed == [(0.0, 1.0, "Hello", "Happy"), (2.0, 1.5, "Bye", "Sad")]
    assert any("Invalid subtitle format" in record.message for record in caplog.records)


def test_ass_formatter_writes_header_and_entries(tmp_path: Path):
    output = tmp_path / "example.ass"
    subtitles = [(0.0, 2.0, "Hello", "Happy")]

    ASSFormatter().generate_file(subtitles, output.as_posix())

    content = output.read_text(encoding="utf-8")
    assert content.startswith(ASSFormatter.ASS_HEADER)
    assert "Dialogue: 0,0:00:00.00,0:00:02.00,Default" in content


def test_srt_formatter_entry_format(tmp_path: Path):
    output = tmp_path / "example.srt"
    subtitles = [(0.0, 1.25, "Hello", "Happy"), (1.25, 0.75, "Bye", "Sad")]

    SRTFormatter().generate_file(subtitles, output.as_posix())

    content = output.read_text(encoding="utf-8")
    assert "1\n00:00:00,000 --> 00:00:01,250\nHello (Happy)" in content
    assert "2\n00:00:01,250 --> 00:00:02,000\nBye (Sad)" in content


def test_vtt_formatter_includes_header_and_arrows(tmp_path: Path):
    output = tmp_path / "example.vtt"
    subtitles = [(0.0, 1.0, "Hello", "Happy")]

    VTTFormatter().generate_file(subtitles, output.as_posix())

    content = output.read_text(encoding="utf-8")
    assert content.startswith("WEBVTT\n\n")
    assert "00:00:00.000 --> 00:00:01.000" in content
