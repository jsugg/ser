"""Speech Emotion Recognition package."""

from importlib.metadata import PackageNotFoundError, version

from .domain import EmotionSegment, TimelineEntry, TranscriptWord

try:
    __version__: str = version("ser")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = ["TranscriptWord", "EmotionSegment", "TimelineEntry", "__version__"]
