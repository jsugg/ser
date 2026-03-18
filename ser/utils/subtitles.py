"""Helpers for exporting timeline rows as subtitle artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from ser.config import TimelineConfig
from ser.domain import TimelineEntry
from ser.utils.logger import get_logger

logger = get_logger(__name__)

type SubtitleFormat = Literal["ass", "srt", "vtt"]
SUPPORTED_SUBTITLE_FORMATS: tuple[SubtitleFormat, ...] = ("ass", "srt", "vtt")
DEFAULT_SUBTITLE_DURATION_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class SubtitleCue:
    """One rendered subtitle cue."""

    start_seconds: float
    end_seconds: float
    text: str
    emotion: str


def infer_subtitle_format(output_path: str) -> SubtitleFormat | None:
    """Infers subtitle format from one output-path suffix."""
    suffix = Path(output_path).suffix.lower().lstrip(".")
    if suffix in SUPPORTED_SUBTITLE_FORMATS:
        return cast(SubtitleFormat, suffix)
    return None


def resolve_subtitle_export_request(
    *,
    output_path: str | None,
    subtitle_format: SubtitleFormat | None,
) -> tuple[SubtitleFormat, str | None] | None:
    """Validates one requested subtitle export and normalizes format resolution."""
    normalized_output_path = output_path.strip() if isinstance(output_path, str) else None
    if isinstance(normalized_output_path, str) and not normalized_output_path:
        raise ValueError("Subtitle output path cannot be empty.")
    if subtitle_format is not None and subtitle_format not in SUPPORTED_SUBTITLE_FORMATS:
        raise ValueError(
            f"Unsupported subtitle format '{subtitle_format}'. " "Expected one of: ass, srt, vtt."
        )
    if subtitle_format is None and normalized_output_path is None:
        return None
    if subtitle_format is not None:
        return subtitle_format, normalized_output_path
    assert isinstance(normalized_output_path, str)
    inferred_format = infer_subtitle_format(normalized_output_path)
    if inferred_format is None:
        raise ValueError(
            "Subtitle export requires --subtitle-format or an output path ending in "
            ".ass, .srt, or .vtt."
        )
    return inferred_format, normalized_output_path


def timeline_to_subtitle_cues(
    timeline: list[TimelineEntry],
    *,
    default_duration_seconds: float = DEFAULT_SUBTITLE_DURATION_SECONDS,
) -> list[SubtitleCue]:
    """Builds subtitle cues from timeline rows with speech content."""
    if default_duration_seconds <= 0.0:
        raise ValueError("default_duration_seconds must be greater than zero.")
    if not timeline:
        return []

    ordered_timeline = sorted(timeline, key=lambda entry: float(entry.timestamp_seconds))
    cues: list[SubtitleCue] = []
    for index, entry in enumerate(ordered_timeline):
        normalized_text = entry.speech.strip()
        if not normalized_text:
            continue
        next_timestamp: float | None = None
        if index + 1 < len(ordered_timeline):
            next_timestamp = float(ordered_timeline[index + 1].timestamp_seconds)
        start_seconds = float(entry.timestamp_seconds)
        if next_timestamp is None or next_timestamp <= start_seconds:
            end_seconds = start_seconds + default_duration_seconds
        else:
            end_seconds = next_timestamp
        cues.append(
            SubtitleCue(
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                text=normalized_text,
                emotion=entry.emotion,
            )
        )
    return cues


def save_timeline_to_subtitles(
    timeline: list[TimelineEntry],
    file_name: str,
    *,
    subtitle_format: SubtitleFormat,
    output_path: str | None = None,
    timeline_config: TimelineConfig | None = None,
) -> str:
    """Writes timeline subtitles and returns the generated artifact path."""
    cues = timeline_to_subtitle_cues(timeline)
    active_config = timeline_config if timeline_config is not None else TimelineConfig()
    target_path = (
        Path(output_path)
        if isinstance(output_path, str) and output_path
        else active_config.folder / f"{Path(file_name).stem}.{subtitle_format}"
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(_render_subtitles(cues, subtitle_format), encoding="utf-8")
    logger.info("Timeline subtitles saved to %s", target_path)
    return str(target_path)


def _render_subtitles(cues: list[SubtitleCue], subtitle_format: SubtitleFormat) -> str:
    """Renders subtitle cues using the requested subtitle format."""
    if subtitle_format == "ass":
        body = "\n".join(_render_ass_entry(cue) for cue in cues)
        return f"{_ASS_HEADER}{body}\n" if body else _ASS_HEADER
    if subtitle_format == "srt":
        body = "\n".join(
            _render_srt_entry(index=index, cue=cue) for index, cue in enumerate(cues, start=1)
        )
        return f"{body}\n" if body else ""
    if subtitle_format == "vtt":
        body = "\n".join(_render_vtt_entry(cue) for cue in cues)
        return f"WEBVTT\n\n{body}\n" if body else "WEBVTT\n"
    raise ValueError(f"Unsupported subtitle format: {subtitle_format}")


def _render_ass_entry(cue: SubtitleCue) -> str:
    """Renders one ASS subtitle line."""
    return (
        "Dialogue: 0,"
        f"{_format_ass_time(cue.start_seconds)},{_format_ass_time(cue.end_seconds)},"
        f"Default,,0,0,0,,{_compose_caption_text(cue)}"
    )


def _render_srt_entry(*, index: int, cue: SubtitleCue) -> str:
    """Renders one SRT subtitle block."""
    return (
        f"{index}\n"
        f"{_format_srt_time(cue.start_seconds)} --> {_format_srt_time(cue.end_seconds)}\n"
        f"{_compose_caption_text(cue)}\n"
    )


def _render_vtt_entry(cue: SubtitleCue) -> str:
    """Renders one WebVTT subtitle block."""
    return (
        f"{_format_vtt_time(cue.start_seconds)} --> {_format_vtt_time(cue.end_seconds)}\n"
        f"{_compose_caption_text(cue)}\n"
    )


def _compose_caption_text(cue: SubtitleCue) -> str:
    """Builds one displayed subtitle payload."""
    normalized_text = cue.text.replace("\r", " ").replace("\n", " ").strip()
    normalized_emotion = cue.emotion.strip()
    if not normalized_emotion:
        return normalized_text
    return f"{normalized_text} ({normalized_emotion})"


def _format_ass_time(seconds: float) -> str:
    """Formats one timestamp for ASS output."""
    total_centiseconds = max(int(round(seconds * 100)), 0)
    hours, remainder = divmod(total_centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def _format_srt_time(seconds: float) -> str:
    """Formats one timestamp for SRT output."""
    total_milliseconds = max(int(round(seconds * 1000)), 0)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def _format_vtt_time(seconds: float) -> str:
    """Formats one timestamp for WebVTT output."""
    total_milliseconds = max(int(round(seconds * 1000)), 0)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


_ASS_HEADER = """[Script Info]
Title: SER Timeline Export
ScriptType: v4.00+
Collisions: Normal
PlayDepth: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0.00,1,1.00,0.00,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
