# Description: Constants used in the project
import os
from typing import List
import torch

DEFAULT_INPUT_VIDEO: str = os.path.join("assets", "input.mp4")

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
# Check to see how much VRAM is available on your GPU and select the model accordingly
# tiny.en: 1.5 GB
# base.en: 2.5 GB
# small.en: 3.5 GB
# medium.en: 5 GB
# large-v1: 8.0 GB
# large-v2: 8.0 GB
# large-v3: 8.0 GB


# Check GPU VRAM and select appropriate model
if torch.cuda.is_available():
    vram_gb = round((torch.cuda.get_device_properties(0).total_memory / 1.073742e+9), 1)
    # print(f"Detected VRAM: {vram_gb} GB")
    if vram_gb >= 8.0:
        MODEL_SIZE = "small.en"
    elif vram_gb >= 5.0:
        MODEL_SIZE = "small.en"
    elif vram_gb >= 3.5:
        MODEL_SIZE = "small.en"
    elif vram_gb >= 2.5:
        MODEL_SIZE = "base.en"
    else:
        MODEL_SIZE = "tiny.en"
else:
    MODEL_SIZE = "tiny.en"  # Fallback if no GPU is available

# print(f"Using model size: {MODEL_SIZE}")
# MODEL_SIZE: str = "large-v3"  # Use this model for transcription
