# Description: Constants used in the project
import os
from typing import Set


# WhisperX supports these models
MODELS_AVAILABLE: Set[str | None] = {
    None,
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
}

# Comprehensive list of video and audio extensions
VIDEO_EXTENSIONS: Set[str] = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".ogv",
    ".3gp",
    ".ts",
}
AUDIO_EXTENSIONS: Set[str] = {
    ".mp3",
    ".mp2",
    ".wav",
    ".flac",
    ".aac",
    ".ogg",
    ".opus",
    ".m4a",
    ".wma",
    ".aiff",
    ".alac",
    ".amr",
}
MEDIA_EXTENSIONS: Set[str] = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS
