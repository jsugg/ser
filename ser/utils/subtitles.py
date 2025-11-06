import argparse
import logging
from abc import ABC, abstractmethod

from ser.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

class SubtitleFormatter(ABC):
    """Abstract base class for subtitle formatters."""

    @abstractmethod
    def format_time(self, seconds: float) -> str:
        """Convert time in seconds to formatted time string."""
        pass

    @abstractmethod
    def generate_entry(self, index: int, start: float, end: float, text: str, emotion: str) -> str:
        """Generate a single subtitle entry."""
        pass

    @abstractmethod
    def generate_file(self, subtitles: list[tuple[float, float, str, str]], output_file: str) -> None:
        """Generate a subtitle file from a list of subtitles."""
        pass

class ASSFormatter(SubtitleFormatter):
    """Formatter for ASS subtitles."""

    ASS_HEADER: str = """
[Script Info]
Title: Generated ASS File
ScriptType: v4.00+
Collisions: Normal
PlayDepth: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0.00,1,1.00,0.00,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def format_time(self, seconds: float) -> str:
        """Convert time in seconds to ASS formatted time string."""
        hours: int = int(seconds // 3600)
        minutes: int = int((seconds % 3600) // 60)
        secs: int = int(seconds % 60)
        millis: int = int((seconds - int(seconds)) * 100)
        return f"{hours:01d}:{minutes:02d}:{secs:02d}.{millis:02d}"

    def generate_entry(self, index: int, start: float, end: float, text: str, emotion: str) -> str:
        """Generate a single ASS subtitle entry."""
        start_time: str = self.format_time(start)
        end_time: str = self.format_time(end)
        logger.debug(
            "ASS Entry: Start %s, End %s, Text %s, Emotion %s",
            start_time,
            end_time,
            text,
            emotion,
        )
        return f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text} ({emotion})"

    def generate_file(self, subtitles: list[tuple[float, float, str, str]], output_file: str) -> None:
        """Generate an ASS file from a list of subtitles."""
        logger.info("Generating ASS file: %s", output_file)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(self.ASS_HEADER)
            for i, (start, duration, text, emotion) in enumerate(subtitles, 1):
                end: float = start + duration
                entry: str = self.generate_entry(i, start, end, text, emotion)
                f.write(entry + '\n')
        logger.info("ASS file generated successfully: %s", output_file)

class SRTFormatter(SubtitleFormatter):
    """Formatter for SRT subtitles."""

    def format_time(self, seconds: float) -> str:
        """Convert time in seconds to SRT formatted time string."""
        hours: int = int(seconds // 3600)
        minutes: int = int((seconds % 3600) // 60)
        secs: int = int(seconds % 60)
        millis: int = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def generate_entry(self, index: int, start: float, end: float, text: str, emotion: str) -> str:
        """Generate a single SRT subtitle entry."""
        start_time: str = self.format_time(start)
        end_time: str = self.format_time(end)
        logger.debug(
            "SRT Entry: Start %s, End %s, Text %s, Emotion %s",
            start_time,
            end_time,
            text,
            emotion,
        )
        return f"{index}\n{start_time} --> {end_time}\n{text} ({emotion})\n"

    def generate_file(self, subtitles: list[tuple[float, float, str, str]], output_file: str) -> None:
        """Generate an SRT file from a list of subtitles."""
        logger.info("Generating SRT file: %s", output_file)
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (start, duration, text, emotion) in enumerate(subtitles, 1):
                end: float = start + duration
                entry: str = self.generate_entry(i, start, end, text, emotion)
                f.write(entry + '\n')
        logger.info("SRT file generated successfully: %s", output_file)

class VTTFormatter(SubtitleFormatter):
    """Formatter for WebVTT subtitles."""

    def format_time(self, seconds: float) -> str:
        """Convert time in seconds to WebVTT formatted time string."""
        hours: int = int(seconds // 3600)
        minutes: int = int((seconds % 3600) // 60)
        secs: int = int(seconds % 60)
        millis: int = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def generate_entry(self, index: int, start: float, end: float, text: str, emotion: str) -> str:
        """Generate a single WebVTT subtitle entry."""
        start_time: str = self.format_time(start)
        end_time: str = self.format_time(end)
        logger.debug(
            "VTT Entry: Start %s, End %s, Text %s, Emotion %s",
            start_time,
            end_time,
            text,
            emotion,
        )
        return f"{start_time} --> {end_time}\n{text} ({emotion})\n"

    def generate_file(self, subtitles: list[tuple[float, float, str, str]], output_file: str) -> None:
        """Generate a WebVTT file from a list of subtitles."""
        logger.info("Generating WebVTT file: %s", output_file)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for i, (start, duration, text, emotion) in enumerate(subtitles, 1):
                end: float = start + duration
                entry: str = self.generate_entry(i, start, end, text, emotion)
                f.write(entry + '\n')
        logger.info("WebVTT file generated successfully: %s", output_file)


DEFAULT_SUBTITLE_DURATION = 1.0


def timeline_to_subtitles(
    timeline: list[tuple[float, str, str]],
    default_duration: float = DEFAULT_SUBTITLE_DURATION,
) -> list[tuple[float, float, str, str]]:
    """Convert a timeline of transcript/emotion entries to subtitle tuples."""
    if not timeline:
        logger.debug("Received empty timeline for subtitle conversion")
        return []

    sorted_timeline: list[tuple[float, str, str]] = sorted(
        timeline,
        key=lambda entry: entry[0],
    )

    subtitles: list[tuple[float, float, str, str]] = []
    for index, (timestamp, emotion, text) in enumerate(sorted_timeline):
        cleaned_text: str = text.strip()
        if not cleaned_text:
            continue

        next_timestamp: float | None = None
        if index + 1 < len(sorted_timeline):
            next_timestamp = float(sorted_timeline[index + 1][0])

        if next_timestamp is not None:
            duration: float = max(next_timestamp - float(timestamp), 0.0)
            if duration == 0.0:
                duration = default_duration
        else:
            duration = default_duration

        subtitles.append((float(timestamp), duration, cleaned_text, emotion))
        logger.debug(
            "Subtitle entry prepared: Start %s, Duration %s, Text %s, Emotion %s",
            timestamp,
            duration,
            cleaned_text,
            emotion,
        )

    return subtitles


class SubtitleGenerator:
    """Main class to generate subtitle files in different formats."""

    def __init__(self, formatter: SubtitleFormatter) -> None:
        self.formatter: SubtitleFormatter = formatter

    def generate_file(self, subtitles: list[tuple[float, float, str, str]], output_file: str) -> None:
        """Generate a subtitle file using the provided formatter."""
        self.formatter.generate_file(subtitles, output_file)


FORMATTERS: dict[str, SubtitleFormatter] = {
    "ass": ASSFormatter(),
    "srt": SRTFormatter(),
    "vtt": VTTFormatter(),
}

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Generate subtitle files (ASS, SRT, WebVTT) from a list of [start, duration, text, emotion]",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "format",
        type=str,
        choices=tuple(FORMATTERS.keys()),
        help="Output subtitle format. Choices are:\n"
             "  ass: Generate Advanced SubStation Alpha (.ass) file\n"
             "  srt: Generate SubRip Subtitle (.srt) file\n"
             "  vtt: Generate Web Video Text Tracks (.vtt) file"
    )
    parser.add_argument(
        "output", 
        type=str, 
        help="Output subtitle file path"
    )
    parser.add_argument(
        "subtitles", 
        type=str, 
        help="List of subtitles in the format: 'start,duration,text,emotion' separated by semicolons. Example:\n"
             "  '0.0,5.0,Hello,Happy;5.0,5.0,World,Surprised'"
    )
    return parser.parse_args()

def parse_subtitles(subtitles_str: str) -> list[tuple[float, float, str, str]]:
    """Parse the input string of subtitles into a list of tuples."""
    subtitles: list[tuple[float, float, str, str]] = []
    for subtitle in subtitles_str.split(';'):
        try:
            start_str, duration_str, text, emotion = subtitle.split(',')
            start: float = float(start_str)
            duration: float = float(duration_str)
            subtitles.append((start, duration, text, emotion))
            logger.debug(
                "Parsed subtitle: Start %s, Duration %s, Text %s, Emotion %s",
                start,
                duration,
                text,
                emotion,
            )
        except ValueError:
            logger.error("Invalid subtitle format: %s", subtitle)
            continue
    return subtitles

def main() -> None:
    """Main entry point for the CLI."""
    args: argparse.Namespace = parse_arguments()
    subtitles: list[tuple[float, float, str, str]] = parse_subtitles(args.subtitles)

    formatter: SubtitleFormatter = FORMATTERS[args.format]
    generator: SubtitleGenerator = SubtitleGenerator(formatter)
    generator.generate_file(subtitles, args.output)

if __name__ == "__main__":
    main()
