import argparse
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

from ser.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    def generate_file(self, subtitles: List[Tuple[float, float, str, str]], output_file: str) -> None:
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
        logging.debug(f"ASS Entry: Start {start_time}, End {end_time}, Text {text}, Emotion {emotion}")
        return f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text} ({emotion})"

    def generate_file(self, subtitles: List[Tuple[float, float, str, str]], output_file: str) -> None:
        """Generate an ASS file from a list of subtitles."""
        logging.info(f"Generating ASS file: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(self.ASS_HEADER)
            for i, (start, duration, text, emotion) in enumerate(subtitles, 1):
                end: float = start + duration
                entry: str = self.generate_entry(i, start, end, text, emotion)
                f.write(entry + '\n')
        logging.info(f"ASS file generated successfully: {output_file}")

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
        logging.debug(f"SRT Entry: Start {start_time}, End {end_time}, Text {text}, Emotion {emotion}")
        return f"{index}\n{start_time} --> {end_time}\n{text} ({emotion})\n"

    def generate_file(self, subtitles: List[Tuple[float, float, str, str]], output_file: str) -> None:
        """Generate an SRT file from a list of subtitles."""
        logging.info(f"Generating SRT file: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (start, duration, text, emotion) in enumerate(subtitles, 1):
                end: float = start + duration
                entry: str = self.generate_entry(i, start, end, text, emotion)
                f.write(entry + '\n')
        logging.info(f"SRT file generated successfully: {output_file}")

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
        logging.debug(f"VTT Entry: Start {start_time}, End {end_time}, Text {text}, Emotion {emotion}")
        return f"{start_time} --> {end_time}\n{text} ({emotion})\n"

    def generate_file(self, subtitles: List[Tuple[float, float, str, str]], output_file: str) -> None:
        """Generate a WebVTT file from a list of subtitles."""
        logging.info(f"Generating WebVTT file: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for i, (start, duration, text, emotion) in enumerate(subtitles, 1):
                end: float = start + duration
                entry: str = self.generate_entry(i, start, end, text, emotion)
                f.write(entry + '\n')
        logging.info(f"WebVTT file generated successfully: {output_file}")

class SubtitleGenerator:
    """Main class to generate subtitle files in different formats."""

    def __init__(self, formatter: SubtitleFormatter) -> None:
        self.formatter: SubtitleFormatter = formatter

    def generate_file(self, subtitles: List[Tuple[float, float, str, str]], output_file: str) -> None:
        """Generate a subtitle file using the provided formatter."""
        try:
            self.formatter.generate_file(subtitles, output_file)
        except Exception as e:
            logging.error(f"Failed to generate file: {output_file}, error: {e}")

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Generate subtitle files (ASS, SRT, WebVTT) from a list of [start, duration, text, emotion]",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "format", 
        type=str, 
        choices=["ass", "srt", "vtt"], 
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

def parse_subtitles(subtitles_str: str) -> List[Tuple[float, float, str, str]]:
    """Parse the input string of subtitles into a list of tuples."""
    subtitles: List[Tuple[float, float, str, str]] = []
    for subtitle in subtitles_str.split(';'):
        try:
            start_str, duration_str, text, emotion = subtitle.split(',')
            start: float = float(start_str)
            duration: float = float(duration_str)
            subtitles.append((start, duration, text, emotion))
            logging.debug(f"Parsed subtitle: Start {start}, Duration {duration}, Text {text}, Emotion {emotion}")
        except ValueError:
            logging.error(f"Invalid subtitle format: {subtitle}")
            continue
    return subtitles

def main() -> None:
    """Main entry point for the CLI."""
    args: argparse.Namespace = parse_arguments()
    subtitles: List[Tuple[float, float, str, str]] = parse_subtitles(args.subtitles)
    
    formatters: Dict[str, SubtitleFormatter] = {
        "ass": ASSFormatter(),
        "srt": SRTFormatter(),
        "vtt": VTTFormatter()
    }
    
    formatter: SubtitleFormatter = formatters[args.format]
    generator: SubtitleGenerator = SubtitleGenerator(formatter)
    generator.generate_file(subtitles, args.output)

if __name__ == "__main__":
    main()
