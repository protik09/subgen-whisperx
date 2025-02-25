# Description: Constants used in the project
import os
from typing import List

DEFAULT_INPUT_VIDEO: str = os.path.join("assets", "input.mp4")

# WhisperX supports these models
MODELS_AVAILABLE: List[str] = [ None,
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
]
