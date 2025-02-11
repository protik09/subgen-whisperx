# Description: Constants used in the project
import os
from typing import List

DEFAULT_INPUT_VIDEO: str = os.path.join("assets", "input.mp4")

MODEL_SIZE: str = "base.en"

# WhisperX supports these models
MODELS_AVAILABLE: List[str] = [
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
]

# Use this model for transcription
MODEL_SIZE: str = "base.en"
