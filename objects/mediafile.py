from dataclasses import dataclass
from pathlib import Path

# This is basically a struct in python
@dataclass
class MediaFile:
    """
    Represents a valid media file with its properties.

    Attributes:
        path (Path): Path to the media file
        is_audio_only (bool): Whether the file contains only audio
    """
    path: Path
    is_audio_only: bool

    @property
    def name(self) -> str:
        """Get the filename."""
        return self.path.name
