from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from types import Dict, Optional, List


# This is basically a struct in python
@dataclass
class MediaFile:
    """
    Represents a valid media file with its properties.

    Attributes:
        path (Path): Path to the media file
        is_audio (bool): Whether the file contains only audio
    """

    status: Enum
    path: Path
    input_srt_path: Path = None
    extracted_audio_path: Path
    is_audio: bool
    successfully_extracted_audio: bool = False
    lang_audio: str
    lang_sub: str
    raw_transcript: Optional[Dict] = None  # Cache for raw whisperx output or loaded SRT data
    aligned_segments: Optional[List[Dict]]
    post_process_segments: Optional[List[Dict]] # Final segments

    def __str__(self) -> str:
        # Should give output: video_to_be_subtitled.mp4 (video)
        _repr = f"{self.path.name} ({'audio' if self.is_audio else 'video'})"
        return _repr

    def __repr__(self) -> str:
        # Should give output: video_to_be_subtitled.mp4 (video)
        _repr = f"{self.path.name} ({'audio' if self.is_audio else 'video'})"
        return _repr

    # Python dataclasses can have Read-Only properties (You can have write ones too but they're more complex 
    # and at that point just use a class instead of a struct)
    # The @property decorator allows you to use basically Mediafile.name to call it instead of Mediafile.name()
    @property
    def name(self) -> str:
        """Get the filename."""
        return str(self.path.name)

    @property
    def full_path(self) -> str:
        """Get the full filepath as a string"""
        return str(self.path.absolute().name)


@dataclass
class MediaStatus:
    """Enumeration of the status messages while going through the pipeline"""

    MEDIA_VALID_START = auto()
    MEDIA_VALIDATING = auto()
    MEDIA_VALID_END = auto()
    AUDIO_EXTRACT_START = auto()
    AUDIO_EXTRACTION = auto()
    AUDIO_EXTRACT_END = auto()
    TRANSCRIPTION_START = auto()
    TRANSCRIPTION = auto()
    TRANSCRIPTION_END = auto()
    ALIGNMENT_START = auto()
    ALIGNING = auto()
    ALIGNMENT_END = auto()
    POST_PROCESS_START = auto()
    POST_PROCESSING = auto()
    POST_PROCESS_END = auto()
    SRT_WRITE_START = auto()
    SRT_WRITING = auto()
    SRT_WRITE_END = auto()
    MERGE_SRT_START = auto()
    MERGE_SRT = auto()
    MERGE_SRT_END = auto()
