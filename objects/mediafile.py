from dataclasses import dataclass
from pathlib import Path


# This is basically a struct in python
@dataclass
class MediaFile:
    """
    Represents a valid media file with its properties.

    Attributes:
        path (Path): Path to the media file
        is_audio (bool): Whether the file contains only audio
    """

    path: Path
    is_audio: bool
    alignment_only_flag: bool
    successfully_extracted_audio: bool = False
    extracted_audio_path: Path
    
    def __str__(self) -> str:
        # Should give output: video_to_be_subtitled.mp4 (video)
        _repr = f"{self.full_name} ({'audio' if self.is_audio else 'video'})"
        return _repr

    def __repr__(self) -> str:
        # Should give output: video_to_be_subtitled.mp4 (video)
        _repr = f"{self.full_name} ({'audio' if self.is_audio else 'video'})"
        return _repr

    # The @property decorator allows you to use basically Mediafile.name to call it instead of Mediafile.name()
    @property
    def name(self) -> str:
        """Get the filename."""
        return str(self.path.name)

    @property
    def full_path(self) -> str:
        """Get the full filepath as a string"""
        return str(self.path.absolute().name)
