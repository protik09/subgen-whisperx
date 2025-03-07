from dataclasses import dataclass
from pathlib import Path

# Python's version of a C struct
@dataclass
class AudioFile:
    """
    Represents an extracted audio file with its properties.

    Attributes:
        path (Path): Path to the audio file
        source_media_path (Path): Path to the original media file
        is_original_audio (bool): Whether this is the original audio (no extraction needed)
    """

    path: Path
    source_media_path: Path
    is_original_audio: bool
