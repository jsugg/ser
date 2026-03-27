"""Tests for subtitle export helpers."""

from pathlib import Path

import pytest

from ser.config import TimelineConfig
from ser.domain import TimelineEntry
from ser.utils.subtitles import (
    infer_subtitle_format,
    resolve_subtitle_export_request,
    save_timeline_to_subtitles,
    timeline_to_subtitle_cues,
)


def test_resolve_subtitle_export_request_infers_format_from_output_path() -> None:
    """Output-path suffix should infer one missing subtitle format."""
    resolved = resolve_subtitle_export_request(
        output_path="exports/sample.vtt",
        subtitle_format=None,
    )

    assert resolved == ("vtt", "exports/sample.vtt")


def test_resolve_subtitle_export_request_rejects_unknown_suffix_without_format() -> None:
    """Unknown subtitle suffix should fail fast without explicit format."""
    with pytest.raises(ValueError, match="Subtitle export requires"):
        resolve_subtitle_export_request(
            output_path="exports/sample.txt",
            subtitle_format=None,
        )


def test_infer_subtitle_format_returns_none_for_unknown_suffix() -> None:
    """Suffix inference should return None for unsupported formats."""
    assert infer_subtitle_format("sample.txt") is None


def test_timeline_to_subtitle_cues_sorts_and_computes_durations() -> None:
    """Cue generation should sort timeline rows and derive end times from neighbors."""
    timeline = [
        TimelineEntry(5.0, "happy", "Third"),
        TimelineEntry(1.0, "neutral", "First"),
        TimelineEntry(3.0, "sad", "Second"),
    ]

    cues = timeline_to_subtitle_cues(timeline, default_duration_seconds=2.0)

    assert [(cue.start_seconds, cue.end_seconds) for cue in cues] == [
        (1.0, 3.0),
        (3.0, 5.0),
        (5.0, 7.0),
    ]


def test_timeline_to_subtitle_cues_skips_blank_text() -> None:
    """Rows without visible speech should not emit subtitle cues."""
    timeline = [
        TimelineEntry(0.0, "happy", "  Hello "),
        TimelineEntry(1.0, "sad", "   "),
        TimelineEntry(2.0, "angry", "World"),
    ]

    cues = timeline_to_subtitle_cues(timeline)

    assert [(cue.start_seconds, cue.text, cue.emotion) for cue in cues] == [
        (0.0, "Hello", "happy"),
        (2.0, "World", "angry"),
    ]


def test_save_timeline_to_subtitles_writes_default_vtt_path(tmp_path: Path) -> None:
    """Subtitle save helper should derive default artifact path from timeline config."""
    timeline = [TimelineEntry(0.0, "happy", "Hello world")]

    subtitle_path = save_timeline_to_subtitles(
        timeline,
        "sample.wav",
        subtitle_format="vtt",
        timeline_config=TimelineConfig(folder=tmp_path),
    )

    written = Path(subtitle_path)
    assert written == tmp_path / "sample.vtt"
    content = written.read_text(encoding="utf-8")
    assert content.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:01.000" in content
    assert "Hello world (happy)" in content


def test_save_timeline_to_subtitles_honors_explicit_output_path(tmp_path: Path) -> None:
    """Explicit output path should override default timeline folder placement."""
    timeline = [TimelineEntry(0.0, "calm", "Hello")]
    output_path = tmp_path / "nested" / "custom.srt"

    subtitle_path = save_timeline_to_subtitles(
        timeline,
        "sample.wav",
        subtitle_format="srt",
        output_path=output_path.as_posix(),
    )

    written = Path(subtitle_path)
    assert written == output_path
    content = written.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:01,000" in content
    assert "Hello (calm)" in content
