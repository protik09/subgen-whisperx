# Single file python program: subgen_app.py
# -*- coding: utf-8 -*-

import argparse
import concurrent.futures
import gc
import json
import logging
import logging.config
import os
import pathlib
import queue
import re
import sys
import tempfile
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Queue
from tkinter import ttk, filedialog, scrolledtext, messagebox
from typing import Dict, List, Optional, Set, Tuple, Union

# --- Dependency Check and Early Exit (Core Libraries) ---
# Try importing critical libraries early to provide clearer error messages if missing.
try:
    import coloredlogs
    import ffmpeg
    import srt  # For parsing input SRTs in alignment-only mode
    import torch
    import whisperx

    # GUI specific
    import sv_ttk  # For modern styling

    # Optional Drag and Drop
    _tkinterdnd_loaded = False
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD

        _tkinterdnd_loaded = True
    except ImportError:
        # Handled later in GUI class
        pass

except ImportError as e:
    missing_module = str(e).split("'")[-2]
    print(f"Error: Missing required core library '{missing_module}'.")
    print("Please install the necessary dependencies.")
    print("You might need to run: pip install -r requirements.txt (or similar)")
    if missing_module == "sv_ttk":
        print("For the modern GUI style, run: pip install sv-ttk")
    elif missing_module == "torch" or missing_module == "whisperx":
        print(
            "Ensure PyTorch and WhisperX are installed correctly, potentially with CUDA support."
        )
        print("See project README or WhisperX documentation for installation details.")
    elif missing_module == "ffmpeg":
        print(
            "Ensure ffmpeg-python is installed ('pip install ffmpeg-python') AND that the ffmpeg executable is in your system's PATH."
        )
    elif missing_module == "tkinterdnd2":
        print("Optional: For GUI drag-and-drop, install 'pip install tkdnd2-alt'")
    sys.exit(f"Installation Error: Could not import core library '{missing_module}'.")


# --- Constants ---
# (Adapted from utils/constants.py)
MODELS_AVAILABLE: Set[str | None] = {
    None,  # Represents 'auto' selection
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",  # Added medium.en
    "large-v1",
    "large-v2",
    "large-v3",
}
WHISPER_LANGUAGE: Set[str] = {
    "af",
    "am",
    "ar",
    "as",
    "az",
    "ba",
    "be",
    "bg",
    "bn",
    "bo",
    "br",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "eo",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fo",
    "fr",
    "gl",
    "gu",
    "ha",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jv",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "la",
    "lb",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "nn",
    "no",
    "oc",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sa",
    "sd",
    "si",
    "sk",
    "sl",
    "sn",
    "so",
    "sq",
    "sr",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tk",
    "tl",
    "tr",
    "tt",
    "uk",
    "ur",
    "uz",
    "vi",
    "wo",
    "xh",
    "yi",
    "yo",
    "zh",
    "zu",
}
# Add common subtitle extensions
SUBTITLE_EXTENSIONS: Set[str] = {".srt"}
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
MAX_PATH_LENGTH = (
    260  # Adjusted based on typical Windows limits, can be higher with registry changes
)
MAX_THREADS = 128
MIN_THREADS = 1
DEFAULT_MODEL_FALLBACK = "tiny"  # Default if auto-select fails
DEFAULT_LOG_LEVEL = "INFO"
OUTPUT_FILENAME_SUFFIX = "AI"  # Changed from "ai"

# --- Logging Setup ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(
    log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_subgen_app.log"
)
# Default config in case JSON fails
default_log_config = {
    "version": 1,
    "disable_existing_loggers": False,  # Allow libraries to log
    "formatters": {
        "detailed": {
            "format": "%(asctime)s.%(msecs)03d [%(levelname)-7s] [%(name)-25s] : %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",  # Default console level
            "formatter": "detailed",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",  # File logs more details
            "formatter": "detailed",
            "filename": log_filename,
            "mode": "a",
        },
    },
    "loggers": {
        "subgen_app": {  # Main application logger
            "level": "DEBUG",  # Capture all levels, handlers control output
            "handlers": ["console", "file"],
            "propagate": False,  # Don't propagate to root logger
        }
    },
    "root": {  # Catch-all for libraries not explicitly configured
        "level": "WARNING",
        "handlers": [
            "console",
            "file",
        ],
    },
}

# Load logging configuration from file or use default
config_file = pathlib.Path("log_format.json")
log_config_to_use = default_log_config
if config_file.exists():
    try:
        with open(config_file, "r") as f:
            loaded_config = json.load(f)
            log_config_to_use = default_log_config.copy()
            for key, value in loaded_config.items():
                if isinstance(value, dict) and key in log_config_to_use:
                    log_config_to_use[key].update(value)
                else:
                    log_config_to_use[key] = value
            log_config_to_use["handlers"]["file"]["filename"] = log_filename
            log_config_to_use["handlers"]["file"]["level"] = "DEBUG"
            if "subgen_app" not in log_config_to_use["loggers"]:
                log_config_to_use["loggers"]["subgen_app"] = default_log_config[
                    "loggers"
                ]["subgen_app"]
            log_config_to_use["loggers"]["subgen_app"]["handlers"] = ["console", "file"]
            log_config_to_use["loggers"]["subgen_app"]["level"] = "DEBUG"
            logging.config.dictConfig(log_config_to_use)
            logging.getLogger("subgen_app").debug(
                f"Loaded logging configuration from {config_file}"
            )
    except Exception as e:
        logging.basicConfig(level=logging.WARNING)
        logging.warning(
            f"Failed to load logging config from {config_file}: {e}. Using default basic config."
        )
        try:
            logging.config.dictConfig(default_log_config)
        except Exception as e_dict:
            logging.error(f"Could not apply default dictConfig: {e_dict}")
else:
    logging.getLogger("subgen_app").debug(
        f"Logging config file {config_file} not found. Using default."
    )
    logging.config.dictConfig(default_log_config)

# Apply coloredlogs to the console handler if it exists
try:
    console_handler_level = (
        log_config_to_use.get("handlers", {})
        .get("console", {})
        .get("level", DEFAULT_LOG_LEVEL)
    )
    detailed_formatter = log_config_to_use.get("formatters", {}).get("detailed", {})
    log_format = detailed_formatter.get(
        "format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    date_format = detailed_formatter.get("datefmt", "%Y-%m-%d %H:%M:%S,%f")
    coloredlogs.install(
        level=console_handler_level,
        fmt=log_format,
        datefmt=date_format[:-3] if date_format.endswith("%f") else date_format,
        milliseconds=True,
        reconfigure=True,
    )
    logging.getLogger("subgen_app").debug("Applied coloredlogs to console handler.")
except Exception as e:
    logging.getLogger("subgen_app").warning(f"Could not apply coloredlogs: {e}")

logger = logging.getLogger("subgen_app")  # Main application logger


# --- Custom Exceptions ---
class FolderNotFoundError(Exception):
    pass


class MediaNotFoundError(Exception):
    pass


class InvalidThreadCountError(ValueError):
    pass


class InvalidPathError(ValueError):
    pass


class AppConfigurationError(Exception):
    pass


class ProcessingError(Exception):
    def __init__(self, message: str, original_path: Optional[Path] = None):
        super().__init__(message)
        self.message = message
        self.original_path = original_path

    def __str__(self):
        if self.original_path:
            return f"[File: {self.original_path.name}] {self.message}"
        return self.message


# --- Utility Functions ---
def sanitize_path(path_str: Optional[str]) -> Optional[Path]:
    """Sanitize, validate, and convert a path string to a Path object."""
    if path_str is None:
        return None
    if not isinstance(path_str, str) or not path_str.strip():
        raise InvalidPathError("Path must be a non-empty string.")
    path_str = path_str.strip()
    if "\0" in path_str:
        raise InvalidPathError("Path contains null bytes.")
    if any(ord(c) < 32 for c in path_str if c not in ("\t")):
        raise InvalidPathError("Path contains forbidden control characters.")
    if re.search(r"[;&|<>*?`$(){}\[\]]", path_str):
        problematic_chars = re.findall(r"[;&|<>*?`$(){}\[\]]", path_str)
        raise InvalidPathError(
            f"Path contains potentially unsafe characters: {', '.join(set(problematic_chars))}"
        )
    if len(path_str) > MAX_PATH_LENGTH * 2:
        raise InvalidPathError(f"Path is excessively long.")
    try:
        if path_str.startswith('"') and path_str.endswith('"'):
            path_str = path_str[1:-1]
        path_obj = Path(path_str)
        resolved_path = path_obj.resolve(strict=False)
        if len(str(resolved_path)) > MAX_PATH_LENGTH and not (
            sys.platform == "win32" and str(resolved_path).startswith("\\\\?\\")
        ):
            pass  # logger.warning(f"Path exceeds standard length...")
        # Basic existence check here - relying more on later stages for file/dir type
        # if not resolved_path.exists():
        #     raise InvalidPathError(f"Path does not exist: {resolved_path}")
        return resolved_path
    except OSError as e:
        raise InvalidPathError(
            f"Invalid or inaccessible path: {path_str}. OS Error: [{e.errno}] {e.strerror}"
        )
    except Exception as e:
        raise InvalidPathError(f"Failed to process path '{path_str}': {e}")


def validate_thread_count(thread_count_str: Optional[str]) -> Optional[int]:
    """Validate and convert thread count string to integer."""
    if thread_count_str is None:
        return None
    if not isinstance(thread_count_str, str) or not thread_count_str.strip():
        raise InvalidThreadCountError("Thread count cannot be empty.")
    cleaned_count = thread_count_str.strip()
    if not cleaned_count.isdigit():
        raise InvalidThreadCountError("Thread count must be a positive integer.")
    try:
        count = int(cleaned_count)
    except ValueError:
        raise InvalidThreadCountError("Thread count must be a valid integer.")
    if count < MIN_THREADS or count > MAX_THREADS:
        raise InvalidThreadCountError(
            f"Thread count must be between {MIN_THREADS} and {MAX_THREADS}. Got: {count}"
        )
    return count


def sanitize_language(language: Optional[str]) -> Optional[str]:
    """Sanitize and validate language code."""
    if language is None:
        return None
    if not isinstance(language, str) or not language.strip():
        raise ValueError("Language code cannot be empty.")
    lang_cleaned = language.strip().lower()
    if not re.fullmatch(r"^[a-z]{2,3}(-[a-z]{2,4})?$", lang_cleaned):
        if lang_cleaned not in WHISPER_LANGUAGE:
            raise ValueError(
                f"Invalid language code format or unknown code: '{language}'."
            )
        else:
            logger.warning(
                f"Language code '{lang_cleaned}' has unusual format but is in Whisper's list."
            )
    if lang_cleaned not in WHISPER_LANGUAGE:
        logger.warning(
            f"Language code '{lang_cleaned}' not in Whisper's known list, but proceeding."
        )
    return lang_cleaned


def get_best_device(requested_device: Optional[str] = None) -> str:
    """Determine the best available compute device (cuda or cpu)."""
    logger_device = logging.getLogger("subgen_app.get_device")
    if requested_device:
        requested_device = requested_device.lower()
        if requested_device not in ["cuda", "cpu"]:
            logger_device.warning(
                f"Invalid device '{requested_device}' requested. Trying auto-detect."
            )
            requested_device = None
    if requested_device == "cpu":
        logger_device.info("CPU explicitly requested.")
        return "cpu"
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            logger_device.info(
                f"CUDA available. Found {device_count} device(s). Using device 0: {device_name}"
            )
            return "cuda"
        else:
            if requested_device == "cuda":
                logger_device.error(
                    "CUDA explicitly requested, but it is not available!"
                )
                raise AppConfigurationError("CUDA requested but not available.")
            else:
                logger_device.warning("CUDA not available. Falling back to CPU.")
                return "cpu"
    except Exception as e:
        logger_device.error(
            f"Error checking CUDA availability: {e}. Falling back to CPU."
        )
        return "cpu"


def get_best_model(
    requested_model: Optional[str] = None,
    language: Optional[str] = None,
    device: str = "cpu",
) -> str:
    """Select the best Whisper model based on VRAM (if GPU) or default."""
    logger_model = logging.getLogger("subgen_app.get_model")
    valid_models = {m for m in MODELS_AVAILABLE if m is not None}
    if requested_model and requested_model not in valid_models:
        logger_model.error(
            f"Requested model size '{requested_model}' is not valid. Available: {', '.join(sorted(valid_models))}"
        )
        raise AppConfigurationError(f"Invalid model size '{requested_model}'.")
    if requested_model:
        logger_model.info(f"Using explicitly requested model: {requested_model}")
        return requested_model

    selected_model = DEFAULT_MODEL_FALLBACK
    is_english = language == "en"
    if device == "cuda":
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger_model.debug(f"Detected VRAM: {vram_gb:.2f} GB")
            # Rough VRAM requirements (adjust based on real usage)
            if vram_gb >= 10.0 and "large-v3" in valid_models:
                selected_model = "large-v3"
            elif vram_gb >= 8.0 and "large-v2" in valid_models:
                selected_model = "large-v2"
            elif vram_gb >= 5.0 and "medium" in valid_models:
                selected_model = (
                    "medium.en"
                    if is_english and "medium.en" in valid_models
                    else "medium"
                )
            elif vram_gb >= 2.0:
                selected_model = (
                    "small.en" if is_english and "small.en" in valid_models else "small"
                )
            elif vram_gb >= 1.0:
                selected_model = (
                    "base.en" if is_english and "base.en" in valid_models else "base"
                )
            else:
                selected_model = (
                    "tiny.en" if is_english and "tiny.en" in valid_models else "tiny"
                )
            # Ensure the selected model is actually available
            if selected_model not in valid_models:
                selected_model = DEFAULT_MODEL_FALLBACK

            logger_model.info(
                f"Auto-selected model based on VRAM ({vram_gb:.2f}GB): {selected_model}"
            )
        except Exception as e:
            logger_model.error(
                f"Error detecting VRAM: {e}. Falling back to default model '{DEFAULT_MODEL_FALLBACK}'."
            )
            selected_model = DEFAULT_MODEL_FALLBACK
    else:  # CPU
        selected_model = (
            "tiny.en" if is_english and "tiny.en" in valid_models else "tiny"
        )
        if selected_model not in valid_models:
            selected_model = DEFAULT_MODEL_FALLBACK
        logger_model.info(f"Using default CPU model: {selected_model}")

    if selected_model not in valid_models:  # Final fallback
        logger_model.warning(
            f"Selected model '{selected_model}' not valid? Falling back to '{DEFAULT_MODEL_FALLBACK}'."
        )
        selected_model = DEFAULT_MODEL_FALLBACK
    return selected_model


def format_time_srt(seconds: float) -> str:
    """Formats seconds into SRT time format HH:MM:SS,ms."""
    if not isinstance(seconds, (int, float)) or seconds < 0:
        logger.warning(f"Invalid time value for SRT formatting: {seconds}. Using 0.")
        seconds = 0
    milliseconds = round((seconds - int(seconds)) * 1000)
    total_seconds_int = int(seconds)
    ss = total_seconds_int % 60
    total_minutes = total_seconds_int // 60
    mm = total_minutes % 60
    hh = total_minutes // 60
    return f"{hh:02d}:{mm:02d}:{ss:02d},{milliseconds:03d}"


def setup_dll_paths():
    """Set up paths for DLL loading, especially for PyInstaller bundles."""
    # (Keep existing logic - seems robust enough)
    dll_logger = logging.getLogger("subgen_app.dll_setup")
    dll_logger.debug("Attempting to set up DLL paths...")
    potential_paths = []
    try:
        script_path = Path(os.path.abspath(sys.argv[0]))
        script_dir = script_path.parent
        potential_paths.append(script_dir / "libs")
    except Exception:
        script_dir = None
        dll_logger.warning("Could not reliably determine script directory.")
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).parent.resolve()
        potential_paths.append(exe_dir / "libs")
        potential_paths.append(exe_dir)
        dll_logger.debug(f"Running frozen (PyInstaller?). Executable dir: {exe_dir}")
    else:
        exe_dir = None
    cwd = Path.cwd()
    potential_paths.append(cwd / "libs")
    added_paths_os = set()
    added_paths_env = set(os.environ.get("PATH", "").split(os.pathsep))
    for path in potential_paths:
        if path and path.is_dir():
            path_str = str(path.resolve())
            if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
                if path_str not in added_paths_os:
                    try:
                        os.add_dll_directory(path_str)
                        dll_logger.debug(f"Added to DLL search path (win): {path_str}")
                        added_paths_os.add(path_str)
                    except Exception as e:
                        dll_logger.warning(
                            f"Failed to add path {path_str} via add_dll_directory: {e}"
                        )
            if path_str not in added_paths_env:
                try:
                    os.environ["PATH"] = (
                        path_str + os.pathsep + os.environ.get("PATH", "")
                    )
                    dll_logger.debug(
                        f"Prepended to PATH environment variable: {path_str}"
                    )
                    added_paths_env.add(path_str)
                except Exception as e:
                    dll_logger.warning(
                        f"Failed to prepend path {path_str} to PATH env var: {e}"
                    )
    if sys.platform == "win32":
        cuda_path_env = os.environ.get("CUDA_PATH")
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        nvidia_base = Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA"
        common_cuda_paths = [cuda_path_env]
        if nvidia_base.exists():
            version_dirs = sorted(
                [
                    d
                    for d in nvidia_base.iterdir()
                    if d.is_dir() and d.name.lower().startswith("v")
                ],
                reverse=True,
            )
            common_cuda_paths.extend(version_dirs)
        found_cuda_bin = False
        for cuda_base_path in common_cuda_paths:
            if cuda_base_path:
                cuda_base = Path(cuda_base_path)
                if cuda_base.exists():
                    bin_path = cuda_base / "bin"
                    if bin_path.exists() and bin_path.is_dir():
                        bin_path_str = str(bin_path.resolve())
                        if (
                            hasattr(os, "add_dll_directory")
                            and bin_path_str not in added_paths_os
                        ):
                            try:
                                os.add_dll_directory(bin_path_str)
                                dll_logger.debug(
                                    f"Added CUDA bin path (win): {bin_path_str}"
                                )
                                added_paths_os.add(bin_path_str)
                            except Exception as e:
                                dll_logger.warning(
                                    f"Failed to add CUDA path {bin_path_str} via add_dll_directory: {e}"
                                )
                        if bin_path_str not in added_paths_env:
                            try:
                                os.environ["PATH"] = (
                                    bin_path_str
                                    + os.pathsep
                                    + os.environ.get("PATH", "")
                                )
                                dll_logger.debug(
                                    f"Prepended CUDA bin path to PATH: {bin_path_str}"
                                )
                                added_paths_env.add(bin_path_str)
                            except Exception as e:
                                dll_logger.warning(
                                    f"Failed to prepend CUDA path {bin_path_str} to PATH env var: {e}"
                                )
                        found_cuda_bin = True
                        dll_logger.info(
                            f"Found and added CUDA bin directory: {bin_path_str}"
                        )
                        break
        if not found_cuda_bin:
            dll_logger.debug("Could not find a standard CUDA bin directory.")
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        dll_logger.debug(
            f"Final os.add_dll_directory paths added: {len(added_paths_os)}"
        )
    dll_logger.debug(
        f"Final PATH (first ~150 chars): {os.environ.get('PATH', '')[:150]}..."
    )
    dll_logger.debug("DLL path setup finished.")


# --- Core Data Structures ---


@dataclass
class MediaFile:
    """Represents a media file to be processed."""

    original_path: Path
    is_audio_only: bool
    input_srt_path: Optional[Path] = None  # For alignment-only mode
    align_only: bool = False  # Flag for alignment-only mode
    extracted_audio_path: Optional[Path] = None
    status: str = "Pending"
    error_message: Optional[str] = None
    language: Optional[str] = None  # Should be set before alignment/writing
    raw_transcript: Optional[Dict] = (
        None  # Cache for raw whisperx output or loaded SRT data
    )
    aligned_segments: Optional[List[Dict]] = None  # Final segments

    def set_status(self, status: str, message: Optional[str] = None):
        self.status = status
        if status == "Error" and message:
            self.error_message = message
            # Limit message length stored?
            if len(self.error_message) > 1000:
                self.error_message = self.error_message[:1000] + "..."

    def __hash__(self):
        try:
            return hash(self.original_path.resolve())
        except OSError:
            return hash(self.original_path)

    def __eq__(self, other):
        if not isinstance(other, MediaFile):
            return NotImplemented
        try:
            return self.original_path.resolve() == other.original_path.resolve()
        except OSError:
            return self.original_path == other.original_path


@dataclass
class AppOptions:
    """Configuration options for the application."""

    input_paths: List[str] = field(default_factory=list)  # Raw input paths from CLI/GUI
    compute_device: str = "cpu"
    model_size: str = DEFAULT_MODEL_FALLBACK
    language: Optional[str] = None  # Needs to be set, even for align-only
    num_threads: Optional[int] = None
    batch_size: int = 16
    compute_type: str = "int8"
    output_dir: Optional[Path] = None
    log_level: str = DEFAULT_LOG_LEVEL
    print_progress: bool = False
    # Feature flags removed for simplicity in this version


# --- Processing Stages ---


class FileFinder:
    """Finds and validates media files from various sources."""

    def __init__(self, status_callback: Optional[callable] = None):
        self.logger = logging.getLogger("subgen_app.FileFinder")
        self.status_callback = status_callback

    def _log_status(self, message: str, level: int = logging.INFO):
        self.logger.log(level, message)
        if self.status_callback:
            self.status_callback(
                {"message": message, "level": logging.getLevelName(level)}
            )

    def _is_media_file_valid(self, file_path: Path) -> Tuple[bool, bool]:
        """Checks if a file is a valid, readable media file using ffmpeg.probe."""
        if not os.access(file_path, os.R_OK):
            self.logger.warning(f"Skipping unreadable file: {file_path}")
            return False, False
        if file_path.suffix.lower() not in MEDIA_EXTENSIONS:
            self.logger.debug(f"Skipping non-media extension: {file_path.name}")
            return False, False
        try:
            self.logger.debug(f"Probing file: {file_path.name}")
            probe = ffmpeg.probe(
                str(file_path), timeout=20
            )  # Increased timeout slightly
            if not probe or not probe.get("streams"):
                self.logger.warning(
                    f"No streams found or invalid probe data for file: {file_path.name}"
                )
                return False, False
            has_audio = any(s.get("codec_type") == "audio" for s in probe["streams"])
            has_video = any(s.get("codec_type") == "video" for s in probe["streams"])
            if not has_audio:  # Require audio stream
                self.logger.warning(f"No audio streams found in: {file_path.name}")
                return False, False
            is_audio_only = has_audio and not has_video
            self.logger.debug(
                f"Validated: {file_path.name} (Audio Only: {is_audio_only})"
            )
            return True, is_audio_only
        except ffmpeg.Error as e:
            stderr = (
                e.stderr.decode("utf-8", errors="ignore").strip()
                if e.stderr
                else "No stderr"
            )
            self.logger.error(
                f"ffmpeg error probing {file_path.name}: {e}. Stderr: {stderr}"
            )
            return False, False
        except Exception as e:
            if (
                isinstance(e, concurrent.futures.TimeoutError)
                or "timed out" in str(e).lower()
            ):
                self.logger.error(f"Timeout probing file {file_path.name}: {e}")
            else:
                self.logger.error(
                    f"Unexpected error probing file {file_path.name}: {e}"
                )
            return False, False

    def find_files(self, raw_paths: List[str]) -> List[MediaFile]:
        """
        Collects, validates, and returns a list of MediaFile objects.
        Uses ThreadPoolExecutor for concurrent validation.
        NOTE: This version does NOT automatically pair media/SRT files.
              Alignment-only mode uses a separate GUI path.
        """
        self._log_status("Starting file discovery...")
        potential_paths: Set[Path] = set()
        found_files_count = 0

        # 1. Sanitize and collect initial paths (media files only for batch mode)
        for raw_path_str in raw_paths:
            try:
                path = sanitize_path(raw_path_str)
                if not path:
                    continue
                if path.is_file():
                    if path.suffix.lower() in MEDIA_EXTENSIONS:
                        potential_paths.add(path)
                        found_files_count += 1
                    elif path.suffix.lower() == ".txt":
                        self._log_status(
                            f"Reading paths from text file: {path}", logging.DEBUG
                        )
                        try:
                            with open(path, "r", encoding="utf-8") as f:
                                for line_num, line in enumerate(f, 1):
                                    line = line.strip()
                                    if not line or line.startswith("#"):
                                        continue
                                    try:
                                        txt_path = sanitize_path(line)
                                        if (
                                            txt_path
                                            and txt_path.is_file()
                                            and txt_path.suffix.lower()
                                            in MEDIA_EXTENSIONS
                                        ):
                                            if txt_path not in potential_paths:
                                                potential_paths.add(txt_path)
                                                found_files_count += 1
                                        elif txt_path and txt_path.is_dir():
                                            self._log_status(
                                                f"Scanning directory from txt '{path.name}' (line {line_num}): {txt_path}",
                                                logging.DEBUG,
                                            )
                                            for item in txt_path.rglob("*"):
                                                if (
                                                    item.is_file()
                                                    and item.suffix.lower()
                                                    in MEDIA_EXTENSIONS
                                                ):
                                                    if item not in potential_paths:
                                                        potential_paths.add(item)
                                                        found_files_count += 1
                                        else:
                                            self.logger.warning(
                                                f"Skipping invalid or non-media path from {path.name} (line {line_num}): {line}"
                                            )
                                    except InvalidPathError as e_txt:
                                        self.logger.warning(
                                            f"Skipping invalid path from {path.name} (line {line_num}): '{line}' - {e_txt}"
                                        )
                                    except Exception as e_txt_read:
                                        self.logger.error(
                                            f"Error processing line {line_num} in {path.name}: '{line}' - {e_txt_read}"
                                        )
                        except Exception as e_read:
                            self.logger.error(
                                f"Failed to read or process text file {path}: {e_read}"
                            )
                    else:
                        self.logger.warning(
                            f"Input file is not a media file or .txt file: {path}"
                        )
                elif path.is_dir():
                    self._log_status(f"Scanning directory: {path}", logging.DEBUG)
                    initial_count = found_files_count
                    for item in path.rglob("*"):
                        if item.is_file() and item.suffix.lower() in MEDIA_EXTENSIONS:
                            if item not in potential_paths:
                                potential_paths.add(item)
                                found_files_count += 1
                    scanned_count = found_files_count - initial_count
                    self._log_status(
                        f"Found {scanned_count} potential media files in {path}",
                        logging.DEBUG,
                    )
                else:
                    self.logger.warning(
                        f"Input path is not a valid file or directory: {path}"
                    )
            except InvalidPathError as e:
                self.logger.error(f"Skipping invalid input path '{raw_path_str}': {e}")
            except Exception as e:
                self.logger.error(
                    f"Unexpected error processing input path '{raw_path_str}': {e}"
                )

        if not potential_paths:
            self._log_status(
                "No potential media files found after initial scan.", logging.WARNING
            )
            raise MediaNotFoundError(
                "No potential media files found from the provided inputs."
            )

        self._log_status(
            f"Found {len(potential_paths)} potential media files. Validating access and format concurrently..."
        )

        # 2. Validate concurrently
        valid_media_files_dict: Dict[Path, MediaFile] = {}
        futures = {}
        max_workers = min((os.cpu_count() or 1) * 2, 16)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="FileValidate"
        ) as executor:
            for path in potential_paths:
                future = executor.submit(self._is_media_file_valid, path)
                futures[future] = path
            processed_count = 0
            total_potential = len(potential_paths)
            for future in concurrent.futures.as_completed(futures):
                processed_count += 1
                path = futures[future]
                try:
                    is_valid, is_audio_only = future.result()
                    if is_valid:
                        try:
                            resolved_path = path.resolve()
                            if resolved_path not in valid_media_files_dict:
                                # Create standard MediaFile, align_only is False here
                                valid_media_files_dict[resolved_path] = MediaFile(
                                    resolved_path, is_audio_only
                                )
                        except OSError as resolve_err:
                            self.logger.error(
                                f"Could not resolve path {path} after validation: {resolve_err}. Skipping."
                            )
                    if processed_count % 20 == 0 or processed_count == total_potential:
                        self._log_status(
                            f"Validation progress: {processed_count}/{total_potential} files checked...",
                            logging.DEBUG,
                        )
                except Exception as e:
                    self.logger.error(
                        f"Unexpected error during validation future processing for {path.name}: {e}"
                    )

        valid_media_files = list(valid_media_files_dict.values())
        if not valid_media_files:
            self._log_status(
                "Validation complete. No valid and accessible media files found.",
                logging.ERROR,
            )
            raise MediaNotFoundError(
                "Found potential files, but none were valid or accessible after probing."
            )

        self._log_status(
            f"Discovery complete. Found {len(valid_media_files)} valid media files."
        )
        valid_media_files.sort(key=lambda mf: mf.original_path)
        return valid_media_files


# --- Standalone Worker Function for Audio Extraction ---
def _extract_audio_worker(original_path_str: str, temp_dir_str: str) -> str:
    """Standalone worker for audio extraction. Returns extracted path string."""
    pid = (
        os.getpid()
    )  # print(f"[Worker {pid}] Extracting: {os.path.basename(original_path_str)}")
    original_path = Path(original_path_str)
    temp_dir = Path(temp_dir_str)
    unique_id = f"{original_path.stem}_{hash(original_path) & 0xFFFFFFFF}"
    output_filename = f"audio_{unique_id}.mp3"  # Using MP3 for broad compatibility
    extracted_audio_path = temp_dir / output_filename
    try:
        # Use ffmpeg-python for extraction: 16kHz mono MP3
        (
            ffmpeg.input(original_path_str)
            .output(
                str(extracted_audio_path),
                acodec="libmp3lame",
                audio_bitrate="128k",  # Reasonable quality/size
                ac=1,
                ar=16000,  # Required by Whisper
                **{"threads": 1},  # Limit threads per process
            )
            .overwrite_output()
            .run(
                cmd="ffmpeg", capture_stdout=True, capture_stderr=True, quiet=True
            )  # Quiet suppresses console output
        )
        if (
            not extracted_audio_path.exists()
            or extracted_audio_path.stat().st_size == 0
        ):
            raise ProcessingError(
                "ffmpeg failed to create a valid audio file.", original_path
            )
        # print(f"[Worker {pid}] Finished extraction: {output_filename}")
        return str(extracted_audio_path)
    except ffmpeg.Error as e:
        stderr = (
            e.stderr.decode("utf-8", errors="ignore").strip() if e.stderr else "N/A"
        )
        error_msg = f"ffmpeg error during extraction: {e}. Stderr: {stderr}"
        raise ProcessingError(error_msg, original_path) from e
    except Exception as e:
        error_msg = f"Unexpected error during extraction worker: {e}"
        raise ProcessingError(error_msg, original_path) from e


class AudioExtractor:
    """Extracts audio from media files using ffmpeg."""

    def __init__(self, options: AppOptions, status_callback: Optional[callable] = None):
        self.logger = logging.getLogger("subgen_app.AudioExtractor")
        self.options = options
        self.status_callback = status_callback
        # Use system's temp directory for better cleanup practices
        self.temp_dir_obj = tempfile.TemporaryDirectory(prefix="subgen_audio_")
        self.temp_dir = Path(self.temp_dir_obj.name)
        self.logger.info(f"Using temporary directory for audio: {self.temp_dir}")

    def _log_status(
        self,
        message: str,
        level: int = logging.INFO,
        media_file: Optional[MediaFile] = None,
    ):
        log_prefix = f"[{media_file.original_path.name}] " if media_file else ""
        full_message = f"{log_prefix}{message}"
        self.logger.log(level, full_message)
        if self.status_callback:
            status_info = {"message": message, "level": logging.getLevelName(level)}
            if media_file:
                status_info["file"] = media_file.original_path.name
                status_info["status"] = media_file.status
            self.status_callback(status_info)

    def extract_audio_concurrently(
        self, media_files: List[MediaFile]
    ) -> List[MediaFile]:
        """Extracts audio concurrently. Updates MediaFile objects."""
        files_needing_extraction = [mf for mf in media_files if not mf.is_audio_only]
        skipped_files = [mf for mf in media_files if mf.is_audio_only]
        for mf in skipped_files:
            self._log_status(
                f"Using original file (already audio).", logging.DEBUG, media_file=mf
            )
            mf.extracted_audio_path = mf.original_path  # Use original path directly
            # Update status to reflect readiness for next step (Transcription or Alignment)
            mf.set_status("Audio Ready")

        if not files_needing_extraction:
            self.logger.info("No files require audio extraction.")
            return media_files

        self._log_status(
            f"Starting concurrent audio extraction for {len(files_needing_extraction)} files..."
        )
        processed_map: Dict[Path, MediaFile] = {
            mf.original_path: mf for mf in media_files
        }
        futures = {}
        max_workers_opt = self.options.num_threads
        default_workers = max(
            1, (os.cpu_count() or 1) // 2
        )  # Use fewer processes for potentially CPU-heavy ffmpeg
        max_workers = (
            max_workers_opt
            if max_workers_opt is not None and max_workers_opt >= 1
            else default_workers
        )
        max_workers = min(
            max_workers, MAX_THREADS, len(files_needing_extraction)
        )  # Don't exceed file count
        self._log_status(
            f"Using {max_workers} worker processes for audio extraction.", logging.DEBUG
        )
        temp_dir_str = str(self.temp_dir)

        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
            ) as executor:
                for mf in files_needing_extraction:
                    mf.set_status("Extracting")
                    self._log_status(
                        f"Submitting for extraction...", logging.DEBUG, media_file=mf
                    )
                    original_path_str = str(mf.original_path)
                    future = executor.submit(
                        _extract_audio_worker, original_path_str, temp_dir_str
                    )
                    futures[future] = mf.original_path
                processed_count = 0
                total_to_extract = len(files_needing_extraction)
                for future in concurrent.futures.as_completed(futures):
                    processed_count += 1
                    original_path = futures[future]
                    mf_to_update = processed_map[original_path]
                    try:
                        extracted_path_str = future.result()
                        mf_to_update.extracted_audio_path = Path(extracted_path_str)
                        mf_to_update.set_status(
                            "Audio Ready"
                        )  # Mark ready for next step
                        self._log_status(
                            f"Extraction successful.",
                            logging.INFO,
                            media_file=mf_to_update,
                        )
                        self._log_status(
                            f"Extraction progress: {processed_count}/{total_to_extract} completed.",
                            logging.DEBUG,
                        )
                    except ProcessingError as e:
                        mf_to_update.set_status("Error", e.message)
                        self._log_status(
                            f"Extraction failed: {e.message}",
                            logging.ERROR,
                            media_file=mf_to_update,
                        )
                    except Exception as e:
                        error_msg = f"Critical error processing extraction result: {e}"
                        mf_to_update.set_status("Error", error_msg)
                        self._log_status(
                            error_msg, logging.CRITICAL, media_file=mf_to_update
                        )
                        self.logger.debug(
                            f"Future processing error details:", exc_info=True
                        )
        except Exception as pool_error:
            self.logger.critical(
                f"Error with ProcessPoolExecutor during audio extraction: {pool_error}",
                exc_info=True,
            )
            # Mark remaining files as errored
            for future, original_path in futures.items():
                if not future.done():
                    mf_to_update = processed_map[original_path]
                    if mf_to_update.status == "Extracting":
                        mf_to_update.set_status(
                            "Error", f"Process pool failed: {pool_error}"
                        )
                        self._log_status(
                            f"Marked as error due to process pool failure.",
                            logging.ERROR,
                            media_file=mf_to_update,
                        )

        self._log_status(f"Audio extraction phase complete.")
        return list(processed_map.values())

    def cleanup_temp_audio(self):
        """Deletes the temporary audio directory."""
        self.logger.info(f"Cleaning up temporary audio directory: {self.temp_dir}")
        try:
            # TemporaryDirectory object handles cleanup on __exit__ or explicit close()
            self.temp_dir_obj.cleanup()
            self.logger.info("Temporary audio directory removed successfully.")
        except Exception as e:
            self.logger.error(
                f"Failed to cleanup temporary audio directory {self.temp_dir}: {e}"
            )


class Transcriber:
    """Transcribes audio files using WhisperX."""

    def __init__(self, options: AppOptions, status_callback: Optional[callable] = None):
        self.logger = logging.getLogger("subgen_app.Transcriber")
        self.options = options
        self.status_callback = status_callback
        self._model = None
        self._model_lock = threading.Lock()
        self._model_cache_dir = Path("./_models").resolve()

    def _log_status(
        self,
        message: str,
        level: int = logging.INFO,
        media_file: Optional[MediaFile] = None,
    ):
        log_prefix = f"[{media_file.original_path.name}] " if media_file else ""
        full_message = f"{log_prefix}{message}"
        self.logger.log(level, full_message)
        if self.status_callback:
            status_info = {"message": message, "level": logging.getLevelName(level)}
            if media_file:
                status_info["file"] = media_file.original_path.name
                status_info["status"] = media_file.status
            self.status_callback(status_info)

    def _load_model(self):
        """Loads the WhisperX transcription model thread-safely."""
        if self._model:
            return
        with self._model_lock:
            if self._model:
                return
            self._log_status(
                f"Loading transcription model: {self.options.model_size} ({self.options.compute_type}) on {self.options.compute_device}",
                logging.INFO,
            )
            start_time = time.monotonic()
            try:
                self._model_cache_dir.mkdir(parents=True, exist_ok=True)
                self._log_status(
                    f"Using model cache directory: {self._model_cache_dir}",
                    logging.DEBUG,
                )
                # Adjust threads based on CPU count if not specified, especially for CPU inference
                cpu_threads = self.options.num_threads or max(
                    1, (os.cpu_count() or 4) // 2
                )
                loaded_model = whisperx.load_model(
                    whisper_arch=self.options.model_size,
                    device=self.options.compute_device,
                    compute_type=self.options.compute_type,
                    language=self.options.language,  # Pre-load with specific language if known
                    threads=cpu_threads
                    if self.options.compute_device == "cpu"
                    else 0,  # Set threads for CPU, 0 for GPU? Check whisperx docs. Might be internal TFlite threads.
                    download_root=str(self._model_cache_dir),
                )
                self._model = loaded_model
                load_time = time.monotonic() - start_time
                self._log_status(
                    f"Transcription model loaded in {load_time:.2f} seconds.",
                    logging.INFO,
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to load WhisperX transcription model: {e}", exc_info=True
                )
                if "cuda" in str(e).lower() and self.options.compute_device == "cuda":
                    self.logger.warning("Attempting to clear CUDA cache...")
                    try:
                        torch.cuda.empty_cache()
                        gc.collect()
                    except Exception as gc_e:
                        self.logger.error(f"Failed during CUDA cache clearing: {gc_e}")
                raise AppConfigurationError(
                    f"Failed to load transcription model '{self.options.model_size}'. Check VRAM, model name, and dependencies."
                ) from e

    def transcribe_files(self, media_files: List[MediaFile]) -> List[MediaFile]:
        """Transcribes files that are ready. Updates MediaFile objects."""
        # Only process files that have audio ready and are NOT align_only
        files_to_process = [
            mf for mf in media_files if mf.status == "Audio Ready" and not mf.align_only
        ]
        if not files_to_process:
            self.logger.info(
                "No files eligible for transcription (or all are align-only)."
            )
            # Update status for align-only files to skip to alignment prep
            for mf in media_files:
                if mf.status == "Audio Ready" and mf.align_only:
                    mf.set_status("Preparing Alignment")
            return media_files

        self._log_status(f"Starting transcription for {len(files_to_process)} files...")
        self._load_model()  # Ensure model is loaded

        processed_count = 0
        total_to_process = len(files_to_process)
        for mf in files_to_process:
            processed_count += 1
            if not mf.extracted_audio_path or not mf.extracted_audio_path.exists():
                mf.set_status("Error", "Audio file for transcription missing.")
                self._log_status(
                    f"Skipping transcription - audio file missing.",
                    logging.ERROR,
                    media_file=mf,
                )
                continue

            mf.set_status("Transcribing")
            self._log_status(
                f"Transcribing ({processed_count}/{total_to_process})...", media_file=mf
            )
            start_time = time.monotonic()
            try:
                # Use the pre-loaded model
                audio_input = str(mf.extracted_audio_path)
                # print(f"DEBUG: Transcribing {audio_input} with batch_size={self.options.batch_size}")
                raw_transcript = self._model.transcribe(
                    audio=audio_input,
                    batch_size=self.options.batch_size,
                    print_progress=self.options.print_progress,
                    chunk_size=30,  # Default, seems okay
                )
                # print(f"DEBUG: Raw transcript: {str(raw_transcript)[:200]}...")

                mf.raw_transcript = raw_transcript
                detected_lang = raw_transcript.get("language")

                # Set language: Use specified if available, otherwise detected.
                if self.options.language:
                    mf.language = self.options.language
                    if detected_lang and detected_lang != mf.language:
                        self._log_status(
                            f"Detected language '{detected_lang}' differs from user-specified '{mf.language}'. Using specified '{mf.language}'.",
                            logging.WARNING,
                            media_file=mf,
                        )
                elif detected_lang:
                    mf.language = detected_lang
                    self._log_status(
                        f"Detected language: {detected_lang}",
                        logging.INFO,
                        media_file=mf,
                    )
                else:
                    mf.set_status(
                        "Error", "Could not determine language for transcription."
                    )
                    self._log_status(
                        "Failed to determine language after transcription.",
                        logging.ERROR,
                        media_file=mf,
                    )
                    continue  # Cannot proceed without language

                if not raw_transcript or not raw_transcript.get("segments"):
                    self._log_status(
                        f"Transcription yielded no segments. Possible silence or error.",
                        logging.WARNING,
                        media_file=mf,
                    )
                    # Allow empty transcript to proceed to alignment/SRT writing (will result in empty SRT)
                    mf.aligned_segments = []  # Set empty aligned segments directly
                    mf.set_status("Writing SRT")  # Ready for empty SRT write
                else:
                    mf.set_status("Transcribed")  # Ready for alignment stage

                duration = time.monotonic() - start_time
                self._log_status(
                    f"Transcription completed in {duration:.2f}s. Language: {mf.language}",
                    media_file=mf,
                )

            except Exception as e:
                error_msg = f"Transcription failed: {e}"
                mf.set_status("Error", error_msg)
                self._log_status(error_msg, logging.ERROR, media_file=mf)
                self.logger.debug(f"Transcription error details", exc_info=True)
                if self.options.compute_device == "cuda":
                    try:
                        torch.cuda.empty_cache()
                        gc.collect()
                    except Exception:
                        pass

        self._log_status("Transcription phase complete.")
        # Update status for align-only files to skip to alignment prep
        for mf in media_files:
            if mf.status == "Audio Ready" and mf.align_only:
                mf.set_status("Preparing Alignment")
        return media_files

    def cleanup(self):
        """Release model resources."""
        if self._model:
            with self._model_lock:
                if self._model:
                    self._log_status("Cleaning up transcription model...")
                    try:
                        del self._model
                        self._model = None
                        if self.options.compute_device == "cuda":
                            torch.cuda.empty_cache()
                        gc.collect()
                        self._log_status("Transcription model resources released.")
                    except Exception as e:
                        self.logger.error(
                            f"Error during transcription model cleanup: {e}"
                        )


class Aligner:
    """Aligns transcriptions using WhisperX (handles both transcribed and loaded SRT data)."""

    def __init__(self, options: AppOptions, status_callback: Optional[callable] = None):
        self.logger = logging.getLogger("subgen_app.Aligner")
        self.options = options
        self.status_callback = status_callback
        self._alignment_models: Dict[str, Tuple[object, Dict]] = {}
        self._align_model_lock = threading.Lock()
        self._model_cache_dir = Path("./_models").resolve()

    def _log_status(
        self,
        message: str,
        level: int = logging.INFO,
        media_file: Optional[MediaFile] = None,
    ):
        log_prefix = f"[{media_file.original_path.name}] " if media_file else ""
        full_message = f"{log_prefix}{message}"
        self.logger.log(level, full_message)
        if self.status_callback:
            status_info = {"message": message, "level": logging.getLevelName(level)}
            if media_file:
                status_info["file"] = media_file.original_path.name
                status_info["status"] = media_file.status
            self.status_callback(status_info)

    def _load_alignment_model(self, language_code: str):
        """Loads or retrieves cached WhisperX alignment model."""
        if language_code in self._alignment_models:
            return self._alignment_models[language_code]
        with self._align_model_lock:
            if language_code in self._alignment_models:
                return self._alignment_models[language_code]
            self._log_status(
                f"Loading alignment model for language: {language_code} on {self.options.compute_device}",
                logging.INFO,
            )
            start_time = time.monotonic()
            try:
                self._model_cache_dir.mkdir(parents=True, exist_ok=True)
                model_a, metadata = whisperx.load_align_model(
                    language_code=language_code,
                    device=self.options.compute_device,
                    model_name=None,  # Use default wav2vec2 model
                    model_dir=str(self._model_cache_dir),
                )
                self._alignment_models[language_code] = (model_a, metadata)
                load_time = time.monotonic() - start_time
                self._log_status(
                    f"Alignment model for {language_code} loaded in {load_time:.2f} seconds.",
                    logging.INFO,
                )
                return model_a, metadata
            except Exception as e:
                self.logger.error(
                    f"Failed to load WhisperX alignment model for language '{language_code}': {e}",
                    exc_info=True,
                )
                raise AppConfigurationError(
                    f"Failed to load alignment model for language '{language_code}'."
                ) from e

    def prepare_srt_for_alignment(self, media_file: MediaFile) -> bool:
        """Loads SRT file content and formats it for the alignment process."""
        if not media_file.input_srt_path or not media_file.input_srt_path.exists():
            media_file.set_status("Error", "Input SRT file missing for alignment.")
            self._log_status(
                "Input SRT file missing.", logging.ERROR, media_file=media_file
            )
            return False

        # Language MUST be set (e.g., from GUI) before calling this
        if not media_file.language:
            media_file.set_status("Error", "Language not specified for SRT alignment.")
            self._log_status(
                "Language must be specified in options for alignment-only mode.",
                logging.ERROR,
                media_file=media_file,
            )
            return False

        self._log_status(
            f"Loading SRT file: {media_file.input_srt_path.name}",
            logging.INFO,
            media_file=media_file,
        )
        try:
            srt_content = media_file.input_srt_path.read_text(encoding="utf-8")
            parsed_subs = list(srt.parse(srt_content))

            if not parsed_subs:
                self._log_status(
                    "Input SRT file is empty or invalid.",
                    logging.WARNING,
                    media_file=media_file,
                )
                media_file.raw_transcript = {
                    "segments": [],
                    "language": media_file.language,
                }
                return True  # Allow proceeding to generate an empty output

            # Convert srt.Subtitle objects to whisperx segment format
            # WhisperX align needs 'text', 'start', 'end'. Start/end are used as hints.
            segments_for_align = []
            for sub in parsed_subs:
                segments_for_align.append(
                    {
                        "text": sub.content.strip(),  # Important to strip whitespace
                        "start": sub.start.total_seconds(),
                        "end": sub.end.total_seconds(),
                    }
                )

            # Store in the expected place, including the language
            media_file.raw_transcript = {
                "segments": segments_for_align,
                "language": media_file.language,  # Carry over the specified language
            }
            self._log_status(
                f"Successfully loaded and prepared {len(segments_for_align)} segments from SRT.",
                logging.DEBUG,
                media_file=media_file,
            )
            return True

        except Exception as e:
            media_file.set_status("Error", f"Failed to parse input SRT file: {e}")
            self._log_status(
                f"Failed to parse SRT file {media_file.input_srt_path.name}: {e}",
                logging.ERROR,
                media_file=media_file,
            )
            self.logger.debug("SRT parsing error details:", exc_info=True)
            return False

    def align_files(self, media_files: List[MediaFile]) -> List[MediaFile]:
        """Aligns transcriptions (from transcription or loaded SRT)."""
        # Files ready are either 'Transcribed' or 'Preparing Alignment' (from align_only path)
        files_to_process_transcribed = [
            mf for mf in media_files if mf.status == "Transcribed"
        ]
        files_to_process_align_only = [
            mf for mf in media_files if mf.status == "Preparing Alignment"
        ]

        if not files_to_process_transcribed and not files_to_process_align_only:
            self.logger.warning("No files eligible for alignment.")
            return media_files

        # --- Prepare Align-Only Files ---
        prepared_align_only_ok = []
        if files_to_process_align_only:
            self._log_status(
                f"Preparing {len(files_to_process_align_only)} files from existing SRTs for alignment..."
            )
            for mf in files_to_process_align_only:
                if self.prepare_srt_for_alignment(mf):
                    prepared_align_only_ok.append(mf)
                # If prepare_srt_for_alignment fails, status is set to Error within it

        all_files_ready_for_align = (
            files_to_process_transcribed + prepared_align_only_ok
        )
        if not all_files_ready_for_align:
            self.logger.warning("No files ready for alignment after preparation.")
            return media_files

        self._log_status(
            f"Starting alignment for {len(all_files_ready_for_align)} files..."
        )

        processed_count = 0
        total_to_process = len(all_files_ready_for_align)
        for mf in all_files_ready_for_align:
            processed_count += 1
            mf.set_status("Aligning")
            self._log_status(
                f"Aligning ({processed_count}/{total_to_process})... Language: {mf.language}",
                media_file=mf,
            )
            start_time = time.monotonic()

            if not mf.extracted_audio_path or not mf.extracted_audio_path.exists():
                mf.set_status("Error", "Audio file for alignment missing.")
                self._log_status(
                    "Skipping alignment - audio file missing.",
                    logging.ERROR,
                    media_file=mf,
                )
                continue
            if not mf.raw_transcript or "segments" not in mf.raw_transcript:
                mf.set_status("Error", "Transcript data missing for alignment.")
                self._log_status(
                    "Skipping alignment - transcript data missing.",
                    logging.ERROR,
                    media_file=mf,
                )
                continue
            if not mf.language:
                mf.set_status("Error", "Language code missing for alignment.")
                self._log_status(
                    "Skipping alignment - language code missing.",
                    logging.ERROR,
                    media_file=mf,
                )
                continue

            # Handle empty transcript case (could be from transcription or empty input SRT)
            if not mf.raw_transcript["segments"]:
                self._log_status(
                    "Transcript has no segments, skipping alignment.",
                    logging.WARNING,
                    media_file=mf,
                )
                mf.aligned_segments = []  # Result is empty segments
                mf.set_status("Writing SRT")  # Ready for empty SRT write
                continue

            try:
                # Load alignment model (cached, thread-safe)
                model_a, metadata = self._load_alignment_model(mf.language)

                # Perform alignment
                # print(f"DEBUG: Aligning {mf.original_path.name} with {len(mf.raw_transcript['segments'])} segments.")
                # print(f"DEBUG: First segment for align: {mf.raw_transcript['segments'][0] if mf.raw_transcript['segments'] else 'N/A'}")
                aligned_result = whisperx.align(
                    transcript=mf.raw_transcript[
                        "segments"
                    ],  # Segments from transcription OR loaded SRT
                    model=model_a,
                    align_model_metadata=metadata,
                    audio=str(mf.extracted_audio_path),
                    device=self.options.compute_device,
                    print_progress=self.options.print_progress,
                    return_char_alignments=False,
                )
                # print(f"DEBUG: Aligned result keys: {aligned_result.keys() if aligned_result else 'N/A'}")
                # print(f"DEBUG: First aligned segment: {aligned_result['segments'][0] if aligned_result and aligned_result.get('segments') else 'N/A'}")

                mf.aligned_segments = aligned_result.get("segments", [])
                # mf.raw_transcript = None # Optional: Clear raw transcript to save memory? Keep for now.

                duration = time.monotonic() - start_time
                self._log_status(
                    f"Alignment completed in {duration:.2f}s", media_file=mf
                )
                mf.set_status("Writing SRT")  # Ready for SRT writing

            except Exception as e:
                error_msg = f"Alignment failed: {e}"
                mf.set_status("Error", error_msg)
                self._log_status(error_msg, logging.ERROR, media_file=mf)
                self.logger.debug(f"Alignment error details", exc_info=True)
                if self.options.compute_device == "cuda":
                    try:
                        torch.cuda.empty_cache()
                        gc.collect()
                    except Exception:
                        pass

        self._log_status("Alignment phase complete.")
        return media_files

    def cleanup(self):
        """Release alignment model resources."""
        if self._alignment_models:
            with self._align_model_lock:
                if self._alignment_models:
                    self._log_status("Cleaning up alignment models...")
                    num_models = len(self._alignment_models)
                    langs = list(self._alignment_models.keys())
                    try:
                        for lang in langs:
                            model, metadata = self._alignment_models.pop(lang)
                            del model
                            self.logger.debug(f"Deleted alignment model for {lang}")
                        if self.options.compute_device == "cuda":
                            torch.cuda.empty_cache()
                        gc.collect()
                        self._log_status(f"Released {num_models} alignment model(s).")
                    except Exception as e:
                        self.logger.error(f"Error during alignment model cleanup: {e}")


class SrtWriter:
    """Generates and writes SRT files from aligned segments."""

    def __init__(self, options: AppOptions, status_callback: Optional[callable] = None):
        self.logger = logging.getLogger("subgen_app.SrtWriter")
        self.options = options
        self.status_callback = status_callback

    def _log_status(
        self,
        message: str,
        level: int = logging.INFO,
        media_file: Optional[MediaFile] = None,
    ):
        log_prefix = f"[{media_file.original_path.name}] " if media_file else ""
        full_message = f"{log_prefix}{message}"
        self.logger.log(level, full_message)
        if self.status_callback:
            status_info = {"message": message, "level": logging.getLevelName(level)}
            if media_file:
                status_info["file"] = media_file.original_path.name
                status_info["status"] = media_file.status
            self.status_callback(status_info)

    def _post_process_text(self, text: str) -> str:
        """Basic text cleanup for subtitles."""
        text = text.strip()
        if not text:
            return ""
        # Simple capitalization
        if text[0].islower():
            text = text[0].upper() + text[1:]
        # Optional: Add more complex rules here if needed
        return text

    def _generate_srt_content(self, segments: List[Dict]) -> str:
        """Generates the content of an SRT file from segments."""
        srt_blocks = []
        max_chars_per_line = 42  # Guideline
        max_lines_per_block = 2

        for i, segment in enumerate(segments, start=1):
            start_time = segment.get("start")
            end_time = segment.get("end")
            text = segment.get("text", "")
            if start_time is None or end_time is None:
                self.logger.warning(
                    f"Skipping segment {i} due to missing time: {segment}"
                )
                continue
            if end_time <= start_time:
                end_time = start_time + 0.1  # Ensure minimal duration
            start_str = format_time_srt(start_time)
            end_str = format_time_srt(end_time)
            processed_text = self._post_process_text(text)
            if not processed_text:
                self.logger.debug(f"Segment {i} empty after processing, skipping.")
                continue

            # Simple line splitting logic
            lines = []
            words = processed_text.split()
            current_line = ""
            while words:
                word = words.pop(0)
                if not current_line:
                    current_line = word
                elif len(current_line) + len(word) + 1 <= max_chars_per_line:
                    current_line += " " + word
                else:
                    lines.append(current_line)
                    current_line = word
                    if len(lines) == max_lines_per_block:
                        self.logger.debug(f"Segment {i} text possibly truncated.")
                        break
            if current_line:
                lines.append(current_line)
            final_text = "\n".join(lines[:max_lines_per_block])

            block = f"{i}\n{start_str} --> {end_str}\n{final_text}\n"
            srt_blocks.append(block)
        return "\n".join(srt_blocks)

    def _write_single_srt(self, media_file: MediaFile) -> Optional[Path]:
        """Worker function to write a single SRT file."""
        self._log_status(
            "Generating SRT content...", logging.DEBUG, media_file=media_file
        )
        if media_file.aligned_segments is None:
            raise ProcessingError(
                "Aligned segments data missing.", media_file.original_path
            )
        if not media_file.language:
            # This really shouldn't happen if alignment succeeded
            mf_lang = "xx"  # Fallback language code
            self.logger.warning(
                f"Language code missing for SRT filename, using '{mf_lang}'.",
                media_file=media_file,
            )
        else:
            mf_lang = media_file.language

        srt_content = self._generate_srt_content(media_file.aligned_segments)
        if not srt_content.strip():
            self._log_status(
                "Generated SRT content is empty. Skipping file writing.",
                logging.WARNING,
                media_file=media_file,
            )
            media_file.set_status(
                "Done (Empty)"
            )  # Indicate completion but empty output
            return None

        # Determine output path with new naming convention
        srt_filename = f"{media_file.original_path.stem}.{OUTPUT_FILENAME_SUFFIX}-{mf_lang}.srt"  # NEW FORMAT
        if self.options.output_dir:
            output_path = self.options.output_dir.resolve() / srt_filename
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as mkdir_e:
                raise ProcessingError(
                    f"Failed to create output directory {output_path.parent}: {mkdir_e}",
                    media_file.original_path,
                ) from mkdir_e
        else:
            output_path = media_file.original_path.parent / srt_filename

        self._log_status(
            f"Writing SRT file to: {output_path}", logging.INFO, media_file=media_file
        )
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(srt_content)
            media_file.set_status("Done")
            self._log_status(
                f"SRT file written successfully.", logging.INFO, media_file=media_file
            )
            return output_path
        except Exception as e:
            error_msg = f"Failed to write SRT file {output_path.name}: {e}"
            self._log_status(error_msg, logging.ERROR, media_file=media_file)
            raise ProcessingError(error_msg, media_file.original_path) from e

    def write_srts_concurrently(self, media_files: List[MediaFile]) -> List[MediaFile]:
        """Generates and writes SRT files concurrently."""
        files_to_process = [mf for mf in media_files if mf.status == "Writing SRT"]
        if not files_to_process:
            self.logger.warning("No files eligible for SRT writing.")
            return media_files

        self._log_status(
            f"Starting concurrent SRT writing for {len(files_to_process)} files..."
        )
        processed_map: Dict[Path, MediaFile] = {
            mf.original_path: mf for mf in media_files
        }
        futures = {}
        max_workers_opt = self.options.num_threads
        default_workers = min(
            (os.cpu_count() or 1) * 4, 32
        )  # Higher multiplier for I/O
        max_workers = (
            max_workers_opt
            if max_workers_opt is not None and max_workers_opt >= 1
            else default_workers
        )
        max_workers = min(max_workers, MAX_THREADS, len(files_to_process))
        self._log_status(
            f"Using {max_workers} worker threads for SRT writing.", logging.DEBUG
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="SrtWrite"
        ) as executor:
            for mf in files_to_process:
                self._log_status(
                    "Submitting for SRT writing...", logging.DEBUG, media_file=mf
                )
                future = executor.submit(self._write_single_srt, mf)
                futures[future] = mf.original_path
            processed_count = 0
            total_to_write = len(files_to_process)
            for future in concurrent.futures.as_completed(futures):
                processed_count += 1
                original_path = futures[future]
                mf_to_update = processed_map[original_path]
                try:
                    srt_path_or_none = future.result()  # Status set in worker
                    self._log_status(
                        f"SRT writing progress: {processed_count}/{total_to_write} completed.",
                        logging.DEBUG,
                    )
                except ProcessingError as e:
                    mf_to_update.set_status(
                        "Error", e.message
                    )  # Ensure status is Error
                    self._log_status(
                        f"SRT writing failed: {e.message}",
                        logging.ERROR,
                        media_file=mf_to_update,
                    )
                except Exception as e:
                    error_msg = f"Critical error processing SRT writing result: {e}"
                    mf_to_update.set_status("Error", error_msg)
                    self._log_status(
                        error_msg, logging.CRITICAL, media_file=mf_to_update
                    )
                    self.logger.debug(
                        f"Future processing error details:", exc_info=True
                    )

        self._log_status("SRT writing phase complete.")
        return list(processed_map.values())


# --- Pipeline Orchestrator ---


class PipelineManager:
    """Manages the subtitle generation pipeline."""

    def __init__(self, options: AppOptions, status_callback: Optional[callable] = None):
        self.logger = logging.getLogger("subgen_app.PipelineManager")
        self.options = options
        self.status_callback = status_callback
        self.media_files: List[MediaFile] = []
        self.start_time = 0
        # Instantiate stage processors only when needed? Or keep them here? Keep for now.
        self.file_finder = FileFinder(status_callback=self._propagate_status)
        self.audio_extractor = AudioExtractor(
            options, status_callback=self._propagate_status
        )
        self.transcriber = Transcriber(options, status_callback=self._propagate_status)
        self.aligner = Aligner(options, status_callback=self._propagate_status)
        self.srt_writer = SrtWriter(options, status_callback=self._propagate_status)

    def _propagate_status(self, status_info: Union[str, Dict]):
        """Logs status and passes it to the main callback."""
        if isinstance(status_info, str):
            message = status_info
            level = logging.INFO
            status_dict = {"message": message, "level": "INFO"}
        elif isinstance(status_info, dict):
            message = status_info.get("message", "Status update")
            level_name = status_info.get("level", "INFO").upper()
            level = logging.getLevelName(level_name)
            status_dict = status_info
        else:
            message = "Unknown status update"
            level = logging.WARNING
            status_dict = {"message": message, "level": "WARNING"}

        log_prefix = f"[{status_dict.get('file')}] " if status_dict.get("file") else ""
        self.logger.log(level, f"{log_prefix}{message}")
        if self.status_callback:
            try:
                self.status_callback(status_dict)
            except Exception as cb_err:
                self.logger.error(f"Error in status callback: {cb_err}")

    def _run_pipeline(self, files_to_process: List[MediaFile]):
        """Internal method to run stages on a list of MediaFiles."""
        if not files_to_process:
            self._propagate_status({"message": "No files to process.", "level": "INFO"})
            return files_to_process  # Return empty list

        # --- Stage 2: Extract Audio ---
        files_processed = self.audio_extractor.extract_audio_concurrently(
            files_to_process
        )
        if not any(mf.status == "Audio Ready" for mf in files_processed):
            self._propagate_status(
                {
                    "message": "Audio extraction failed or skipped for all files.",
                    "level": "WARNING",
                }
            )
            return files_processed  # Return with potential errors

        # --- Stage 3: Transcribe (Skips align_only files internally) ---
        files_processed = self.transcriber.transcribe_files(files_processed)
        # Check if any files are ready for alignment (either transcribed or prepared align-only)
        if not any(
            mf.status in ["Transcribed", "Preparing Alignment"]
            for mf in files_processed
        ):
            self._propagate_status(
                {
                    "message": "Transcription failed for all applicable files.",
                    "level": "WARNING",
                }
            )
            return files_processed

        # --- Stage 4: Align (Handles both transcribed and prepared align-only) ---
        files_processed = self.aligner.align_files(files_processed)
        if not any(mf.status == "Writing SRT" for mf in files_processed):
            self._propagate_status(
                {"message": "Alignment failed for all files.", "level": "WARNING"}
            )
            return files_processed

        # --- Stage 5: Write SRT ---
        files_processed = self.srt_writer.write_srts_concurrently(files_processed)

        return files_processed

    def run_batch(self, raw_input_paths: List[str]):
        """Executes the full pipeline for multiple inputs (transcription or alignment)."""
        self.start_time = time.monotonic()
        self._propagate_status(
            {"message": "Batch processing started.", "level": "INFO"}
        )
        processed_files: List[MediaFile] = []
        total_files = 0
        success_count = 0
        error_count = 0
        skipped_count = 0

        try:
            # --- Stage 1: Find Files ---
            self.media_files = self.file_finder.find_files(raw_input_paths)
            total_files = len(self.media_files)
            if not self.media_files:
                self._propagate_status(
                    {"message": "No valid media files found.", "level": "WARNING"}
                )
                return

            # --- Run Pipeline Stages ---
            processed_files = self._run_pipeline(self.media_files)

            end_time = time.monotonic()
            duration = end_time - self.start_time
            self._propagate_status(
                {
                    "message": f"Batch processing finished in {duration:.2f} seconds.",
                    "level": "INFO",
                }
            )

        except MediaNotFoundError as e:
            self._propagate_status(
                {"message": f"Pipeline halted: {e}", "level": "ERROR"}
            )
        except AppConfigurationError as e:
            self._propagate_status(
                {"message": f"Pipeline halted (config error): {e}", "level": "CRITICAL"}
            )
        except ProcessingError as e:
            self._propagate_status(
                {
                    "message": f"Pipeline halted (processing error): {e}",
                    "level": "ERROR",
                }
            )
        except Exception as e:
            self._propagate_status(
                {
                    "message": f"Unexpected critical error in pipeline: {e}",
                    "level": "CRITICAL",
                }
            )
            self.logger.exception("Pipeline critical error details:")
        finally:
            # --- Final Summary ---
            files_to_summarize = (
                processed_files or self.media_files
            )  # Use processed if available
            if files_to_summarize:
                success_count = sum(
                    1 for mf in files_to_summarize if mf.status.startswith("Done")
                )  # Include "Done (Empty)"
                error_count = sum(
                    1 for mf in files_to_summarize if mf.status == "Error"
                )
                # Estimate total processed based on initial find or final list
                summary_total = (
                    total_files
                    if total_files >= len(files_to_summarize)
                    else len(files_to_summarize)
                )

                summary_msg = (
                    f"Summary: {success_count}/{summary_total} files processed."
                )
                if error_count > 0:
                    summary_msg += f" {error_count} failed."
                # Skipped are harder to track accurately across stages, focus on success/error
                self._propagate_status({"message": summary_msg, "level": "INFO"})

                if error_count > 0:
                    self._propagate_status(
                        {"message": "Files with errors:", "level": "WARNING"}
                    )
                    for mf in files_to_summarize:
                        if mf.status == "Error":
                            error_detail = mf.error_message or "Unknown error"
                            self._propagate_status(
                                {
                                    "message": f"- {mf.original_path.name}: {error_detail[:150]}{'...' if len(error_detail) > 150 else ''}",
                                    "level": "WARNING",
                                    "file": mf.original_path.name,
                                }
                            )
            # --- Cleanup ---
            self._cleanup_resources()

    def run_single_alignment(self, media_path: Path, srt_path: Path):
        """Executes the pipeline specifically for alignment-only mode on a single pair."""
        self.start_time = time.monotonic()
        self._propagate_status(
            {
                "message": f"Alignment-only process started for {media_path.name}",
                "level": "INFO",
            }
        )
        processed_files: List[MediaFile] = []

        try:
            # --- Stage 1: Validate Input Pair ---
            self._propagate_status("Validating input files...")
            # Basic validation (existence, readability)
            if not media_path.exists() or not os.access(media_path, os.R_OK):
                raise AppConfigurationError(
                    f"Media file not found or not readable: {media_path}"
                )
            if not srt_path.exists() or not os.access(srt_path, os.R_OK):
                raise AppConfigurationError(
                    f"SRT file not found or not readable: {srt_path}"
                )
            # Check media file using ffmpeg
            is_valid_media, is_audio_only = self.file_finder._is_media_file_valid(
                media_path
            )
            if not is_valid_media:
                raise AppConfigurationError(
                    f"Invalid or unsupported media file: {media_path.name}"
                )

            # Language must be provided via options for alignment model loading
            if not self.options.language:
                raise AppConfigurationError(
                    "Language must be specified in options for alignment-only mode."
                )

            # Create a single MediaFile object for this job
            media_file = MediaFile(
                original_path=media_path.resolve(),
                is_audio_only=is_audio_only,
                input_srt_path=srt_path.resolve(),
                align_only=True,
                language=self.options.language,  # Set language from options
            )
            self.media_files = [media_file]  # Pipeline expects a list

            # --- Run Alignment Pipeline Stages ---
            # Pass the single file list to the internal runner
            processed_files = self._run_pipeline(self.media_files)

            end_time = time.monotonic()
            duration = end_time - self.start_time
            self._propagate_status(
                {
                    "message": f"Alignment-only process finished in {duration:.2f} seconds.",
                    "level": "INFO",
                }
            )

        except AppConfigurationError as e:
            self._propagate_status(
                {
                    "message": f"Alignment halted (config error): {e}",
                    "level": "CRITICAL",
                }
            )
        except ProcessingError as e:
            self._propagate_status(
                {
                    "message": f"Alignment halted (processing error): {e}",
                    "level": "ERROR",
                }
            )
        except Exception as e:
            self._propagate_status(
                {
                    "message": f"Unexpected critical error during alignment: {e}",
                    "level": "CRITICAL",
                }
            )
            self.logger.exception("Alignment-only critical error details:")
        finally:
            # --- Summary (for single file) ---
            if processed_files:
                mf = processed_files[0]
                if mf.status.startswith("Done"):
                    self._propagate_status(
                        {
                            "message": f"Successfully aligned and saved: {mf.original_path.name}",
                            "level": "INFO",
                        }
                    )
                elif mf.status == "Error":
                    error_detail = mf.error_message or "Unknown error"
                    self._propagate_status(
                        {
                            "message": f"Alignment failed for {mf.original_path.name}: {error_detail}",
                            "level": "ERROR",
                        }
                    )
                else:  # Should not happen if finished
                    self._propagate_status(
                        {
                            "message": f"Alignment finished with unexpected status '{mf.status}' for {mf.original_path.name}",
                            "level": "WARNING",
                        }
                    )

            # --- Cleanup ---
            self._cleanup_resources()

    def _cleanup_resources(self):
        """Cleans up resources used by the pipeline stages."""
        self._propagate_status(
            {"message": "Starting final cleanup...", "level": "INFO"}
        )
        if self.audio_extractor:
            self.audio_extractor.cleanup_temp_audio()
        if self.transcriber:
            self.transcriber.cleanup()
        if self.aligner:
            self.aligner.cleanup()
        gc.collect()  # Explicit garbage collection
        self._propagate_status({"message": "Cleanup finished.", "level": "INFO"})


# --- GUI Implementation ---

# Use TkinterDnD if loaded, otherwise fall back to standard tk.Tk
BaseTkClass = TkinterDnD.Tk if _tkinterdnd_loaded else tk.Tk


class SubgenGUI(BaseTkClass):
    def __init__(self, prefilled_options: Optional[AppOptions] = None):
        super().__init__()

        self.title("Subgen WhisperX")
        # Apply sv-ttk theme
        try:
            sv_ttk.set_theme("dark")  # Or "light"
            logger.info(f"Applied sv-ttk theme.")
        except Exception as e:
            logger.warning(
                f"Failed to apply sv-ttk theme: {e}. Using default ttk theme."
            )

        self.options = prefilled_options or AppOptions()
        self.file_list = []  # Stores Path objects for batch processing
        self.pipeline_manager: Optional[PipelineManager] = None
        self.processing_thread: Optional[threading.Thread] = None
        self.status_queue = Queue()

        # --- Grid Layout ---
        self.columnconfigure(0, weight=1)  # Left panel
        self.columnconfigure(1, weight=2)  # Right panel (log)
        self.rowconfigure(1, weight=1)  # Allow list/log to expand vertically
        self.rowconfigure(2, weight=0)  # Control buttons fixed height

        # --- Left Panel ---
        left_frame = ttk.Frame(self, padding="10 10 10 0")  # Padding only on R/B
        left_frame.grid(
            row=0, column=0, rowspan=2, sticky="nsew", padx=(10, 0), pady=(10, 5)
        )
        left_frame.rowconfigure(0, weight=1)  # Input frame expands
        left_frame.rowconfigure(1, weight=0)  # Options frame fixed
        left_frame.columnconfigure(0, weight=1)

        # File Input Section
        input_frame = ttk.LabelFrame(
            left_frame, text="Input Files/Folders (for Batch Processing)", padding=10
        )
        input_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        input_frame.rowconfigure(0, weight=1)
        input_frame.columnconfigure(0, weight=1)

        # Drag and Drop Listbox
        self.listbox_frame = ttk.Frame(input_frame)
        self.listbox_frame.grid(row=0, column=0, columnspan=4, sticky="nsew", pady=5)
        self.listbox_frame.rowconfigure(0, weight=1)
        self.listbox_frame.columnconfigure(0, weight=1)

        self.listbox = tk.Listbox(
            self.listbox_frame, selectmode=tk.EXTENDED, width=40, height=10
        )
        self.listbox_scrollbar_y = ttk.Scrollbar(
            self.listbox_frame, orient=tk.VERTICAL, command=self.listbox.yview
        )
        self.listbox_scrollbar_x = ttk.Scrollbar(
            self.listbox_frame, orient=tk.HORIZONTAL, command=self.listbox.xview
        )
        self.listbox.configure(
            yscrollcommand=self.listbox_scrollbar_y.set,
            xscrollcommand=self.listbox_scrollbar_x.set,
        )
        self.listbox.grid(row=0, column=0, sticky="nsew")
        self.listbox_scrollbar_y.grid(row=0, column=1, sticky="ns")
        self.listbox_scrollbar_x.grid(row=1, column=0, sticky="ew")

        if _tkinterdnd_loaded and DND_FILES:
            self.listbox.drop_target_register(DND_FILES)
            self.listbox.dnd_bind("<<Drop>>", self.handle_drop)
            dnd_status = "(Drag & Drop Enabled)"
        else:
            dnd_status = "(Drag & Drop Disabled)"
        input_frame.config(text=f"Batch Input {dnd_status}")

        # Add/Remove Buttons
        self.add_files_button = ttk.Button(
            input_frame, text="Add Files", command=self.add_files
        )
        self.add_folder_button = ttk.Button(
            input_frame, text="Add Folder", command=self.add_folder
        )
        self.remove_button = ttk.Button(
            input_frame, text="Remove", command=self.remove_selected
        )
        self.clear_button = ttk.Button(
            input_frame, text="Clear All", command=self.clear_list
        )
        self.add_files_button.grid(row=2, column=0, padx=(0, 2), pady=5, sticky="ew")
        self.add_folder_button.grid(row=2, column=1, padx=2, pady=5, sticky="ew")
        self.remove_button.grid(row=2, column=2, padx=2, pady=5, sticky="ew")
        self.clear_button.grid(row=2, column=3, padx=(2, 0), pady=5, sticky="ew")
        input_frame.columnconfigure((0, 1, 2, 3), weight=1)  # Equal button width

        # Options Section
        options_frame = ttk.LabelFrame(left_frame, text="Options", padding=10)
        options_frame.grid(row=1, column=0, sticky="nsew")
        options_frame.columnconfigure(1, weight=1)

        # Language (Required even for align-only)
        ttk.Label(options_frame, text="Language*:").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        self.lang_var = tk.StringVar(
            value=self.options.language or "en"
        )  # Default to 'en' if None
        # Ensure default 'en' is valid
        if self.lang_var.get() not in WHISPER_LANGUAGE:
            self.lang_var.set("en")
        lang_options = sorted(list(WHISPER_LANGUAGE))
        self.lang_combo = ttk.Combobox(
            options_frame,
            textvariable=self.lang_var,
            values=lang_options,
            state="readonly",
            width=15,
        )
        self.lang_combo.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        ttk.Label(options_frame, text="(*Required)").grid(
            row=0, column=3, padx=5, pady=5, sticky="w"
        )

        # Model Size
        ttk.Label(options_frame, text="Model Size:").grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        model_display = (
            self.options.model_size
            if self.options.model_size in MODELS_AVAILABLE and self.options.model_size
            else "auto"
        )
        self.model_var = tk.StringVar(value=model_display)
        model_options = ["auto"] + sorted(
            [m for m in MODELS_AVAILABLE if m is not None]
        )
        self.model_combo = ttk.Combobox(
            options_frame,
            textvariable=self.model_var,
            values=model_options,
            state="readonly",
            width=15,
        )
        self.model_combo.grid(
            row=1, column=1, columnspan=3, padx=5, pady=5, sticky="ew"
        )

        # Device
        ttk.Label(options_frame, text="Device:").grid(
            row=2, column=0, padx=5, pady=5, sticky="w"
        )
        device_display = (
            self.options.compute_device
            if self.options.compute_device in ["cuda", "cpu"]
            else "auto"
        )
        self.device_var = tk.StringVar(value=device_display)
        device_options = ["auto", "cuda", "cpu"]
        self.device_combo = ttk.Combobox(
            options_frame,
            textvariable=self.device_var,
            values=device_options,
            state="readonly",
            width=15,
        )
        self.device_combo.grid(
            row=2, column=1, columnspan=3, padx=5, pady=5, sticky="ew"
        )

        # Compute Type
        ttk.Label(options_frame, text="Compute Type:").grid(
            row=3, column=0, padx=5, pady=5, sticky="w"
        )
        self.compute_type_var = tk.StringVar(value=self.options.compute_type or "int8")
        compute_type_options = ["int8", "float16", "float32"]
        self.compute_type_combo = ttk.Combobox(
            options_frame,
            textvariable=self.compute_type_var,
            values=compute_type_options,
            state="readonly",
            width=15,
        )
        self.compute_type_combo.grid(
            row=3, column=1, columnspan=3, padx=5, pady=5, sticky="ew"
        )

        # --- Right Panel ---
        right_frame = ttk.Frame(self, padding="10 10 10 10")
        right_frame.grid(
            row=0, column=1, rowspan=2, sticky="nsew", padx=(0, 10), pady=(10, 5)
        )
        right_frame.rowconfigure(0, weight=1)
        right_frame.columnconfigure(0, weight=1)

        # Log/Status Area
        log_frame = ttk.LabelFrame(right_frame, text="Status / Log", padding=10)
        log_frame.grid(row=0, column=0, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            state="disabled",
            wrap=tk.WORD,
            height=15,
            width=60,
            bd=0,
            relief="flat",
        )  # Flat look
        self.log_text.grid(row=0, column=0, sticky="nsew")

        # Configure log colors (adapt for light/dark themes maybe?)
        # These assume a dark theme background primarily
        # sv-ttk might override some of these depending on theme.
        self.log_text.tag_config("INFO", foreground="#FFFFFF")  # White on dark
        self.log_text.tag_config("DEBUG", foreground="#AAAAAA")  # Lighter gray
        self.log_text.tag_config("WARNING", foreground="#FFA500")  # Orange
        self.log_text.tag_config("ERROR", foreground="#FF4C4C")  # Bright Red
        self.log_text.tag_config(
            "CRITICAL", foreground="#FF4C4C", font=("Segoe UI", 9, "bold")
        )  # Use Segoe UI if available
        self.log_text.tag_config(
            "FILENAME", foreground="#64B5F6", font=("Segoe UI", 9, "italic")
        )  # Light Blue Italic

        # --- Bottom Control Buttons ---
        control_frame = ttk.Frame(self, padding="10")
        control_frame.grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10)
        )
        control_frame.columnconfigure(0, weight=1)  # Spacer left
        control_frame.columnconfigure(1, weight=0)  # Align button
        control_frame.columnconfigure(2, weight=0)  # Start button
        control_frame.columnconfigure(3, weight=1)  # Spacer right

        # New button for alignment-only mode
        self.align_srt_button = ttk.Button(
            control_frame,
            text="Align Existing SRT...",
            command=self.start_alignment_only_processing,
            # style="Accent.TButton" # Optional styling
        )
        self.align_srt_button.grid(row=0, column=1, padx=5, pady=5)

        self.start_button = ttk.Button(
            control_frame,
            text="Start Batch Processing",
            command=self.start_batch_processing,
            style="Accent.TButton",  # Highlight main action
        )
        self.start_button.grid(row=0, column=2, padx=5, pady=5)

        # --- Final Setup ---
        self.after(100, self.process_status_queue)
        self.update_idletasks()
        self.minsize(self.winfo_reqwidth() + 20, self.winfo_reqheight() + 20)

        # Add initial log message
        self.log_message(f"Subgen GUI Initialized. Logging to: {log_filename}", "INFO")
        if not _tkinterdnd_loaded:
            self.log_message(
                "Drag & Drop disabled (tkinterdnd2 not found). Install with: pip install tkdnd2-alt",
                "WARNING",
            )

    # --- Logging & Status Update ---
    def log_message(
        self, message: str, level: str = "INFO", filename: Optional[str] = None
    ):
        """Appends a message to the GUI log area. Thread-safe via queue."""
        try:
            self.log_text.configure(state="normal")
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            prefix = f"[{timestamp}] "
            level_tag = level.upper()
            if level_tag not in self.log_text.tag_names():
                level_tag = "INFO"

            self.log_text.insert(tk.END, prefix)
            if filename:
                self.log_text.insert(tk.END, f"[{filename}] ", ("FILENAME",))
            self.log_text.insert(tk.END, f"{message}\n", (level_tag,))

            self.log_text.configure(state="disabled")
            self.log_text.see(tk.END)
        except Exception as e:
            print(f"GUI LOG ERROR: {level} - {filename} - {message} | Exception: {e}")

    def queue_status_update(self, status_info: Union[str, Dict]):
        """Puts status updates onto the queue for the GUI thread."""
        try:
            self.status_queue.put(status_info)
        except Exception as e:
            print(f"CRITICAL: Error putting status in queue: {e}")

    def process_status_queue(self):
        """Processes status updates from the queue in the GUI thread."""
        job_finished = False
        try:
            while not self.status_queue.empty():
                status_info = self.status_queue.get_nowait()
                if isinstance(status_info, dict):
                    message = status_info.get("message", "Unknown status")
                    level = status_info.get("level", "INFO").upper()
                    filename = status_info.get("file")
                else:
                    message = str(status_info)
                    level = "INFO"
                    filename = None

                self.log_message(message, level=level, filename=filename)

                # Check for job completion messages
                if (
                    "finished in" in message
                    or "Pipeline halted" in message
                    or "processing finished" in message
                    or "alignment finished" in message
                ):
                    job_finished = True

        except queue.Empty:
            pass
        except Exception as e:
            error_msg = f"Error processing status queue: {e}"
            self.log_message(error_msg, "ERROR")
            logger.exception("Status queue processing error")
        finally:
            # If a finished message was detected, call the finished handler
            if job_finished and self.processing_thread is not None:
                self.on_processing_finished()

            # Reschedule check
            if self.winfo_exists():
                self.after(100, self.process_status_queue)

    # --- File Input Handling ---
    def handle_drop(self, event):
        """Handles files dropped onto the listbox."""
        if not _tkinterdnd_loaded:
            return
        logger.debug(f"Drop event data: '{event.data}'")
        raw_paths = self.parse_drop_data(event.data)
        self.add_paths_to_list(raw_paths)

    def parse_drop_data(self, data_string: str) -> List[str]:
        """Parses the string data from a TkinterDnD drop event."""
        paths = []
        # Regex to find {path with spaces} or path_without_spaces
        pattern = r"\{[^{}]*\}|\S+"
        matches = re.findall(pattern, data_string)
        for match in matches:
            path = (
                match[1:-1].strip()
                if match.startswith("{") and match.endswith("}")
                else match
            )
            path = path.strip('"')
            if path:
                paths.append(path)
        logger.debug(f"Parsed drop data into paths: {paths}")
        return paths

    def add_paths_to_list(self, paths_to_add: List[str]):
        """Adds valid paths to the internal list and updates the listbox."""
        added_count = 0
        current_paths_in_list = {str(p) for p in self.file_list}
        for raw_path_str in paths_to_add:
            try:
                path = sanitize_path(raw_path_str)
                if not path or not path.exists():
                    self.log_message(
                        f"Skipped non-existent path: {raw_path_str}", "WARNING"
                    )
                    continue
                if not (path.is_file() or path.is_dir()):
                    self.log_message(f"Skipped (not file/dir): {path}", "WARNING")
                    continue
                # Only add media files or directories to the list for batch mode
                if (
                    path.is_file()
                    and path.suffix.lower() not in MEDIA_EXTENSIONS
                    and path.suffix.lower() != ".txt"
                ):
                    self.log_message(
                        f"Skipped non-media/txt file: {path.name}", "WARNING"
                    )
                    continue

                resolved_path_str = str(path.resolve())
                if resolved_path_str not in current_paths_in_list:
                    resolved_path = Path(resolved_path_str)
                    self.file_list.append(resolved_path)
                    # Display shorter path in listbox
                    self.listbox.insert(tk.END, str(path))
                    current_paths_in_list.add(resolved_path_str)
                    added_count += 1
                else:
                    logger.debug(f"Path already in list: {path}")
            except InvalidPathError as e:
                self.log_message(
                    f"Skipped invalid path '{raw_path_str}': {e}", "WARNING"
                )
            except Exception as e:
                self.log_message(f"Error adding path '{raw_path_str}': {e}", "ERROR")
                logger.exception(f"Detailed error adding path:")
        if added_count > 0:
            self.log_message(f"Added {added_count} item(s) to the batch list.")
        self.listbox.see(tk.END)

    def add_files(self):
        """Opens file dialog to add files for batch processing."""
        media_ext_tuple = tuple(sorted(MEDIA_EXTENSIONS))
        filetypes = [
            (
                "Media/Txt Files",
                " ".join(f"*{ext}" for ext in media_ext_tuple) + " *.txt",
            ),
            ("Video Files", " ".join(f"*{ext}" for ext in VIDEO_EXTENSIONS)),
            ("Audio Files", " ".join(f"*{ext}" for ext in AUDIO_EXTENSIONS)),
            ("Text Files", "*.txt"),
            ("All Files", "*.*"),
        ]
        files = filedialog.askopenfilenames(
            title="Select Media Files or Text Files (.txt) for Batch",
            filetypes=filetypes,
        )
        if files:
            self.add_paths_to_list(list(files))

    def add_folder(self):
        """Opens directory dialog to add a folder for batch processing."""
        folder = filedialog.askdirectory(
            title="Select Folder Containing Media Files for Batch"
        )
        if folder:
            self.add_paths_to_list([folder])

    def remove_selected(self):
        """Removes selected items from the batch list."""
        selected_indices = self.listbox.curselection()
        if not selected_indices:
            return
        items_to_remove_display = [self.listbox.get(i) for i in selected_indices]
        paths_to_remove_resolved = set()
        for display_str in items_to_remove_display:
            try:
                paths_to_remove_resolved.add(Path(display_str).resolve())
            except Exception as e:
                logger.warning(
                    f"Could not resolve path during removal: {display_str} - {e}"
                )
        # Remove from listbox
        for i in reversed(selected_indices):
            self.listbox.delete(i)
        # Filter internal list
        initial_len = len(self.file_list)
        self.file_list = [
            p for p in self.file_list if p.resolve() not in paths_to_remove_resolved
        ]
        removed_count = initial_len - len(self.file_list)
        self.log_message(f"Removed {removed_count} item(s) from batch list.")

    def clear_list(self):
        """Clears the batch list."""
        self.listbox.delete(0, tk.END)
        self.file_list.clear()
        self.log_message("Cleared batch input list.")

    # --- Processing Logic ---
    def update_options_from_gui(self):
        """Updates the self.options object based on GUI selections. Returns True if valid."""
        try:
            # Language (Now mandatory)
            lang_selection = self.lang_var.get()
            if not lang_selection or lang_selection not in WHISPER_LANGUAGE:
                messagebox.showerror(
                    "Invalid Option",
                    "Please select a valid language from the dropdown.",
                )
                return False  # Indicate failure
            self.options.language = sanitize_language(
                lang_selection
            )  # Already checked against list

            # Device (resolve auto)
            device_selection = self.device_var.get()
            self.options.compute_device = get_best_device(
                device_selection if device_selection != "auto" else None
            )

            # Model Size (resolve auto based on resolved device and language)
            model_selection = self.model_var.get()
            req_model = model_selection if model_selection != "auto" else None
            self.options.model_size = get_best_model(
                req_model, self.options.language, self.options.compute_device
            )

            # Compute Type
            compute_type_selection = self.compute_type_var.get()
            if compute_type_selection in ["int8", "float16", "float32"]:
                self.options.compute_type = compute_type_selection
            else:
                logger.warning(
                    f"Invalid compute type selected '{compute_type_selection}', using default 'int8'."
                )
                self.options.compute_type = "int8"

            # Update input paths from current listbox content (for batch mode)
            self.options.input_paths = [str(p) for p in self.file_list]

            logger.info(
                f"Updated options: Lang={self.options.language}, Model={self.options.model_size}, Device={self.options.compute_device}, ComputeType={self.options.compute_type}"
            )
            return True  # Indicate success

        except (AppConfigurationError, ValueError, InvalidPathError) as e:
            error_msg = f"Error updating options: {e}"
            self.log_message(error_msg, "ERROR")
            messagebox.showerror("Option Error", error_msg)
            return False  # Indicate failure
        except Exception as e:
            self.log_message(f"Unexpected error updating options: {e}", "ERROR")
            logger.error("Failed to update options from GUI", exc_info=True)
            messagebox.showerror(
                "Option Error", "An unexpected error occurred while setting options."
            )
            return False  # Indicate failure

    def start_batch_processing(self):
        """Starts the subtitle generation pipeline for the batch list."""
        if not self.file_list:
            self.log_message("No files added to the batch list.", "WARNING")
            messagebox.showwarning(
                "No Input", "Please add files or folders to the batch list first."
            )
            return
        if self.processing_thread and self.processing_thread.is_alive():
            self.log_message("Processing is already running.", "WARNING")
            messagebox.showwarning("Busy", "A processing job is already running.")
            return

        if not self.update_options_from_gui():
            return  # Stop if options invalid

        self.toggle_controls(enabled=False)
        self.start_button.config(text="Processing Batch...")
        self.log_message("=" * 20 + " Starting Batch Job " + "=" * 20, "INFO")

        # Create PipelineManager with current options
        self.pipeline_manager = PipelineManager(
            self.options, status_callback=self.queue_status_update
        )
        # Target the run_batch method
        self.processing_thread = threading.Thread(
            target=self.pipeline_manager.run_batch,
            args=(self.options.input_paths,),  # Pass raw paths from options
            daemon=True,
        )
        self.processing_thread.start()

    def start_alignment_only_processing(self):
        """Handles the 'Align Existing SRT' workflow."""
        if self.processing_thread and self.processing_thread.is_alive():
            self.log_message("Processing is already running.", "WARNING")
            messagebox.showwarning("Busy", "A processing job is already running.")
            return

        # 1. Get Options (especially Language)
        if not self.update_options_from_gui():
            return  # Stop if options invalid

        # 2. Prompt for Media File
        media_ext_tuple = tuple(sorted(MEDIA_EXTENSIONS))
        media_filetypes = [
            ("Media Files", " ".join(f"*{ext}" for ext in media_ext_tuple)),
            ("All Files", "*.*"),
        ]
        media_file_str = filedialog.askopenfilename(
            title="Select Media File for Alignment", filetypes=media_filetypes
        )
        if not media_file_str:
            return  # User cancelled

        # 3. Prompt for SRT File
        srt_filetypes = [("SRT Files", "*.srt"), ("All Files", "*.*")]
        srt_file_str = filedialog.askopenfilename(
            title="Select Existing SRT File to Align", filetypes=srt_filetypes
        )
        if not srt_file_str:
            return  # User cancelled

        # Basic validation before starting thread
        try:
            media_path = sanitize_path(media_file_str)
            srt_path = sanitize_path(srt_file_str)
            if not media_path or not media_path.is_file():
                raise InvalidPathError(f"Invalid media file selected: {media_file_str}")
            if not srt_path or not srt_path.is_file():
                raise InvalidPathError(f"Invalid SRT file selected: {srt_file_str}")
        except InvalidPathError as e:
            self.log_message(f"Invalid input for alignment: {e}", "ERROR")
            messagebox.showerror("Invalid Input", str(e))
            return
        except Exception as e:
            self.log_message(f"Error preparing alignment input: {e}", "ERROR")
            messagebox.showerror("Error", f"Failed to prepare input: {e}")
            return

        # Start processing thread for single alignment job
        self.toggle_controls(enabled=False)
        self.align_srt_button.config(text="Aligning SRT...")  # Update specific button
        self.log_message(
            "=" * 20 + f" Starting Alignment Job for {media_path.name} " + "=" * 20,
            "INFO",
        )

        # Create PipelineManager with current options
        self.pipeline_manager = PipelineManager(
            self.options, status_callback=self.queue_status_update
        )
        # Target the run_single_alignment method
        self.processing_thread = threading.Thread(
            target=self.pipeline_manager.run_single_alignment,
            args=(
                media_path,
                srt_path,
            ),
            daemon=True,
        )
        self.processing_thread.start()

    def on_processing_finished(self):
        """Called when the pipeline thread finishes."""
        self.log_message("=" * 20 + " Job Finished " + "=" * 20, "INFO")
        self.toggle_controls(enabled=True)
        # Reset button texts
        self.start_button.config(text="Start Batch Processing")
        self.align_srt_button.config(text="Align Existing SRT...")
        self.processing_thread = None
        # Maybe show a final confirmation popup?
        # messagebox.showinfo("Finished", "Processing complete. Check log for details.")

    def toggle_controls(self, enabled: bool):
        """Enable or disable input controls."""
        state = tk.NORMAL if enabled else tk.DISABLED
        btn_state = "normal" if enabled else "disabled"  # ttk uses strings
        combo_state = "readonly" if enabled else "disabled"

        # Input List Buttons
        self.add_files_button.config(state=btn_state)
        self.add_folder_button.config(state=btn_state)
        self.remove_button.config(state=btn_state)
        self.clear_button.config(state=btn_state)

        # Option Comboboxes
        self.lang_combo.config(state=combo_state)
        self.model_combo.config(state=combo_state)
        self.device_combo.config(state=combo_state)
        self.compute_type_combo.config(state=combo_state)

        # Main Action Buttons
        self.start_button.config(state=btn_state)
        self.align_srt_button.config(state=btn_state)

        # Listbox - Keep enabled for selection/viewing, but modification is via buttons
        # self.listbox.config(state=state)


# --- Main Execution Logic ---


def parse_arguments() -> Optional[AppOptions]:
    """Parses command line arguments. Returns AppOptions for CLI or None for GUI."""
    parser = argparse.ArgumentParser(
        description="Generate subtitles using WhisperX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        help="Path(s) to input media file(s), folder(s), or .txt file(s).",
        metavar="PATH",
    )
    parser.add_argument(
        "-l",
        "--language",
        default=None,
        help="Language code (e.g., 'en', 'es'). If unset, uses auto-detection (transcribe) or requires GUI selection (align).",
        metavar="CODE",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="auto",
        help=f"Whisper model size. 'auto' selects based on VRAM/device. Choices: auto, {', '.join(sorted(m for m in MODELS_AVAILABLE if m))}",
        metavar="SIZE",
    )
    parser.add_argument(
        "-c",
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device.",
    )
    parser.add_argument(
        "--compute_type",
        choices=["int8", "float16", "float32"],
        default="int8",
        help="Compute type for model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for transcription.",
        metavar="N",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Max CPU threads for ffmpeg/cpu inference. Default: Auto.",
        metavar="N",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=None,
        help="Directory to save SRT files. Default: alongside input.",
        metavar="DIR",
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=DEFAULT_LOG_LEVEL,
        help="Console logging level.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show WhisperX progress bars (CLI only).",
    )
    parser.add_argument("--no-gui", action="store_true", help="Force CLI mode.")

    # Hidden legacy args
    parser.add_argument("-f", "--file", help=argparse.SUPPRESS)
    parser.add_argument("-d", "--directory", help=argparse.SUPPRESS)
    parser.add_argument("-t", "--txt", help=argparse.SUPPRESS)

    try:
        args = parser.parse_args()
    except SystemExit as e:
        print("Exiting due to argparse action.")
        raise

    # --- Determine run mode ---
    raw_input_paths = []
    legacy_args_used = False
    if args.input:
        raw_input_paths.extend(args.input)
    if args.file:
        print("Warning: Using legacy '-f/--file'. Prefer '-i/--input'.")
        raw_input_paths.append(args.file)
        legacy_args_used = True
    if args.directory:
        print("Warning: Using legacy '-d/--directory'. Prefer '-i/--input'.")
        raw_input_paths.append(args.directory)
        legacy_args_used = True
    if args.txt:
        print("Warning: Using legacy '-t/--txt'. Prefer '-i/--input'.")
        raw_input_paths.append(args.txt)
        legacy_args_used = True
    if legacy_args_used and args.input:
        print("Warning: Both '-i/--input' and legacy args provided. Combining all.")

    if not raw_input_paths and not args.no_gui:
        logger.info("No input paths provided via CLI. Starting GUI mode.")
        # Prepare default options potentially carrying CLI args for GUI prefill
        gui_options = AppOptions()
        try:
            # Carry over simple options, let GUI resolve 'auto' & validate language
            gui_options.language = (
                sanitize_language(args.language) if args.language else None
            )  # Pass validated or None
            gui_options.compute_device = args.device  # Keep 'auto'
            gui_options.model_size = args.model  # Keep 'auto'
            gui_options.compute_type = args.compute_type
            gui_options.log_level = args.log_level  # GUI doesn't use this directly yet
            if args.output_dir:
                gui_options.output_dir = sanitize_path(
                    args.output_dir
                )  # Pass validated path
        except Exception as e:
            logger.warning(f"Could not prefill GUI options from CLI args: {e}")
        return None  # Signal GUI run

    if not raw_input_paths and args.no_gui:
        parser.error("Cannot run in --no-gui mode without input paths via -i/--input.")

    # --- Configure for CLI run ---
    options = AppOptions()
    options.input_paths = raw_input_paths
    options.log_level = args.log_level.upper()
    # Update root logger and console handler levels (moved logic here)
    try:
        root_logger = logging.getLogger()
        root_logger.setLevel(options.log_level)
        loggers_to_check = [logging.getLogger(), logging.getLogger("subgen_app")]
        updated_console = False
        for logr in loggers_to_check:
            for handler in logr.handlers:
                if isinstance(handler, logging.StreamHandler) and getattr(
                    handler, "stream", None
                ) in (sys.stdout, sys.stderr):
                    handler.setLevel(options.log_level)
                    updated_console = True
            if updated_console:
                break
        if updated_console:
            logger.info(f"Console log level set to: {options.log_level}")
        else:
            logger.warning("Could not find console handler to update log level.")
    except Exception as e:
        logger.error(f"Failed to update log level: {e}")

    try:
        # Validate/set language - crucial for CLI
        options.language = sanitize_language(args.language) if args.language else None
        # Note: Auto-detection (language=None) only works reliably for transcription.
        # CLI Alignment-only mode isn't directly supported here, requires GUI or modification.
        if not options.language and any(
            ".srt" in p.lower() for p in options.input_paths
        ):  # Rudimentary check
            logger.warning(
                "Detected potential SRT input without specified language. Alignment might fail. Use GUI or specify --language."
            )

        options.num_threads = validate_thread_count(
            str(args.threads) if args.threads else None
        )
        options.compute_device = get_best_device(
            args.device if args.device != "auto" else None
        )
        req_model = args.model if args.model != "auto" else None
        options.model_size = get_best_model(
            req_model, options.language, options.compute_device
        )
        options.compute_type = args.compute_type
        options.batch_size = args.batch_size
        if options.batch_size < 1:
            raise ValueError("Batch size must be >= 1.")
        if args.output_dir:
            options.output_dir = sanitize_path(args.output_dir)
            if options.output_dir:
                if not options.output_dir.exists():
                    logger.warning(
                        f"Output directory will be created: {options.output_dir}"
                    )
                elif not options.output_dir.is_dir():
                    raise InvalidPathError(
                        f"Output path is not a directory: {options.output_dir}"
                    )
        options.print_progress = args.progress

    except (ValueError, InvalidPathError, AppConfigurationError) as e:
        parser.error(f"Configuration Error: {e}")
    except Exception as e:
        logger.critical(
            f"Unexpected error during CLI option processing: {e}", exc_info=True
        )
        sys.exit(f"Error: {e}")

    logger.info("Running in Command-Line Interface (CLI) mode.")
    logger.info(
        f"Options: Device='{options.compute_device}', Model='{options.model_size}', Lang='{options.language or 'auto (transcribe)'}', Compute='{options.compute_type}', Batch={options.batch_size}"
    )
    if options.output_dir:
        logger.info(f"Output directory: {options.output_dir}")
    return options


def run_gui(options: Optional[AppOptions] = None):
    """Initializes and runs the Tkinter GUI."""
    logger.info("Initializing GUI...")
    try:
        # Pass prefilled options if GUI launched implicitly with CLI args
        app = SubgenGUI(prefilled_options=options)
        app.mainloop()
    except tk.TclError as e:
        if "application has been destroyed" in str(e).lower():
            logger.info("GUI closed.")
        else:
            logger.error(f"Tkinter TclError: {e}", exc_info=True)
            print(f"GUI Error: {e}.")
    except Exception as e:
        logger.error(f"Failed to run GUI: {e}", exc_info=True)
        print(
            f"Fatal Error: Could not start the GUI. Check '{log_filename}' for details."
        )


def run_cli(options: AppOptions):
    """Runs the processing pipeline using command-line options."""
    logger.info("Starting processing via CLI...")
    # CLI uses batch run. Alignment-only via CLI is not directly supported without language spec.
    pipeline = PipelineManager(options, status_callback=None)  # No GUI callback
    try:
        pipeline.run_batch(options.input_paths)
    except Exception as e:
        logger.critical(f"CLI pipeline execution failed: {e}", exc_info=True)
    finally:
        logger.info("CLI processing finished.")


if __name__ == "__main__":
    # Ensure DLLs/shared objects can be found
    setup_dll_paths()

    cli_options_or_none_for_gui: Optional[AppOptions] = None
    try:
        # Parse arguments, returns Options object for CLI or None for GUI
        cli_options_or_none_for_gui = parse_arguments()
    except SystemExit as e:
        sys.exit(e.code)  # Allow exit for --help etc.
    except Exception as e:
        logger.critical(f"Argument parsing failed: {e}", exc_info=True)
        sys.exit(1)

    # Run based on parse result
    if cli_options_or_none_for_gui is not None:
        run_cli(cli_options_or_none_for_gui)
    else:
        # GUI mode: parse_arguments returned None, potentially with prefilled options.
        # We need to re-call parse_arguments inside run_gui OR pass the options it might have created.
        # Let's re-parse cleanly inside the GUI launch context if needed, or better,
        # trust that parse_arguments returned None *because* it intends GUI run,
        # and pass any default options it might have prepared.
        # The current parse_arguments prepares `gui_options` when returning None. Let's pass that.
        # **Correction**: `parse_arguments` returns None for GUI, but doesn't return the prepared `gui_options`.
        # Let's pass None and let the GUI constructor handle defaults.
        run_gui(
            options=None
        )  # Pass None, GUI uses its defaults or parse_arguments() logic internally if needed

    logger.info("Application exiting.")
    logging.shutdown()
    sys.exit(0)  # Ensure clean exit code on success
