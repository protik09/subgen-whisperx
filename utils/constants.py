# Description: Constants used in the project
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

WHISPER_LANGUAGE: Set[str] = {
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca",
    "cs", "cy", "da", "de", "el", "en", "eo", "es", "et", "eu", "fa", "fi", "fo",
    "fr", "gl", "gu", "ha", "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it",
    "ja", "jv", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo", "lt", "lv",
    "mg", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn", "no", "oc",
    "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so",
    "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt",
    "uk", "ur", "uz", "vi", "wo", "xh", "yi", "yo", "zh", "zu"
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
