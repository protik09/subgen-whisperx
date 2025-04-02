import sys
import os
import argparse
import logging
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import utils.sanitize as sanitize
from utils.exceptions import (
    FolderNotFoundError,
)
from utils.constants import MODELS_AVAILABLE

# Maximum allowed path length (Windows has 260 char limit by default)
MAX_PATH_LENGTH = 260
# Maximum reasonable thread count
MAX_THREADS = 128
# Minimum reasonable thread count
MIN_THREADS = 1


@dataclass
class AppOptions:
    """Configuration options for the subtitle generator."""

    input_file: Optional[str] = None
    input_directory: Optional[str] = None
    input_language: Optional[list[str]] = None
    output_directory: Optional[Path]
    valid_input_file_paths: list[str]
    compute_device: Optional[str] = None    # "cpu", "cuda"
    model_size: Optional[str] = None
    log_level: str = "INFO"
    num_threads: Optional[int] = None       # Max threads to use
    print_progress_flag: bool = False       # Flag for WhisperX progress bar
    txt: Optional[str] = None               # Txt file containing file paths to media files
    cli_flag: Optional[bool] = False
    merge_sub_to_file: Optional[bool] = None

    def __str__(self):
        """
        Return a string representation of the Options object.

        Returns:
            str: A string representation of the Options object containing its properties
                    and their current values.
        """
        # Normalize paths for consistent representation
        file_path = os.path.normpath(str(self.file)) if self.file else None
        in_dir_path = os.path.normpath(str(self.input_directory)) if self.directory else None
        out_dir_path = os.path.normpath(str(self.output_directory)) if self.output_directory else None
        txt_path = os.path.normpath(str(self.txt)) if self.txt else None

        args_str: str = f"""Input File: {file_path} | Input Directory: {in_dir_path} | Output Directory: {out_dir_path} | Compute Device: {self.compute_device} | Model Selected: {self.model_size} | Log Level: {self.log_level} | Media Language: {self.language} | Number of Threads: {self.num_threads} | Input Media Paths File: {txt_path} | Print Progress Flag: {self.print_progress_flag} | CLI Flag: {self.cli_flag} | Merge Subtitles to File: {self.merge_sub_to_file}"""
        return args_str


def parse_arguments() -> AppOptions:
    """
    Parse and validate command line arguments.

    Returns:
        Options: Validated configuration options

    Raises:
        Various exceptions for invalid inputs
    """
    # Init options object
    options = AppOptions()
    parser = argparse.ArgumentParser(description="Subtitle Generator")
    parser.add_argument(
        "-f",
        "--file",
        default=None,
        help="Path to the input media file",
    )
    parser.add_argument(
        "-d",
        "--directory",
        default=None,
        help="Path to directory containing media files",
    )
    parser.add_argument(
        "-c",
        "--compute_device",
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use for computation (cuda or cpu), automatically selects cuda if available",
    )
    parser.add_argument(
        "-m",
        "--model_size",
        default=None,
        choices=MODELS_AVAILABLE,
        help="Whisper model size to use for transcription (default: auto-select based on VRAM)",
    )
    parser.add_argument(
        "-log",
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: ERROR)",
    )
    parser.add_argument(
        "-l",
        "--language",
        default=None,
        help="Set the language for subtitles",
    )
    parser.add_argument(
        "-n",
        "--num_threads",
        default=None,
        help="Set the number of threads for transcription",
    )
    parser.add_argument(
        "-t",
        "--txt",
        default=None,
        help="Pass a txt file containing the paths to either media files or directories",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Flag to start the GUI",
    )

    logger = logging.getLogger("options")
    try:
        args = parser.parse_args()
    except Exception as e:
        logger.error(f"Error parsing arguments: {str(e)}")
        raise
    # Set up logging first
    options.log_level = args.log_level
    logger.setLevel(options.log_level)

    # Sanitize and validate all inputs
    options.file = args.file
    options.directory = args.directory
    options.txt = sanitize.sanitize_path(args.txt)
    options.compute_device = (
        args.compute_device if args.compute_device else _get_device(options)
    )
    options.model_size = args.model_size if args.model_size else _get_model(options)
    options.language = sanitize.sanitize_language(args.language)
    options.num_threads = sanitize.validate_thread_count(args.num_threads)

    # If no args are passed to argparser, print help and exit
    if (options.file is None) and (options.directory is None) and (options.txt is None):
        parser.print_help()
        sys.exit(0)

    # If log level is less than INFO, set the progress bar to True
    # Note: This is abusing the logging.getlevelname function and may break
    options.print_progress_flag = (
        True
        if logging.getLevelName(options.log_level) < logging.getLevelName("INFO")
        else False
    )

    # Validate existence of paths after sanitization
    if options.directory and not os.path.isdir(options.directory):
        logger.error(f"Error: Directory '{options.directory}' does not exist.")
        raise FolderNotFoundError(message=f"Directory '{options.directory}' does not exist.")

    if options.file and not os.path.isfile(options.file):
        logger.error(f"Error: File '{options.file}' does not exist.")
        raise FileNotFoundError(f"File '{options.file}' does not exist.")

    if options.txt and not os.path.isfile(options.txt):
        logger.error(f"Error: Text file '{options.txt}' does not exist.")
        raise FileNotFoundError(f"Text file '{options.txt}' does not exist.")
    return options


def _get_device(options: AppOptions) -> str:
    """
    Determine the best available device with graceful fallback to CPU.
    Returns:
        str: The device to use for computation.
    """
    from torch import cuda  # Here for performance reasons

    _logger = logging.getLogger("options_get_device")

    if options.compute_device is None or "cuda" in options.compute_device.lower():
        try:
            if cuda.is_available():
                _logger.info("CUDA available. Using GPU acceleration.")
                return "cuda"
            else:
                _logger.warning("CUDA not available, falling back to CPU")
        except Exception as e:
            _logger.error(f"Warning: Error checking CUDA availability ({str(e)})")
            _logger.warning("Falling back to CPU.")
    else:
        pass
    return "cpu"


def _get_model(options: AppOptions) -> str:
    """
    Get the Whisper model size to use for transcription.
    Returns:
        str: The Whisper model size to use for transcription.
    """
    from torch import cuda  # Here for performance reasons

    _logger = logging.getLogger("options_get_model")

    if options.model_size not in MODELS_AVAILABLE:
        _logger.error(f"Model size '{options.model_size}' is not available.")
        raise ValueError(
            f"Model size '{options.model_size}' is not valid. Available models: {MODELS_AVAILABLE}"
        )
    if options.model_size is None:
        # Check to see how much VRAM is available on your GPU and select the model accordingly
        if cuda.is_available():
            vram_gb = round(
                (cuda.get_device_properties(0).total_memory / 1.073742e9), 1
            )
            _logger.debug(f"Detected VRAM: {vram_gb} GB")
            if vram_gb >= 9.0:
                options.model_size = "large-v2"
            elif vram_gb >= 7.5:
                options.model_size = "medium"
            elif vram_gb >= 4.5:
                options.model_size = "small.en" if options.language == "en" else "small"
            elif vram_gb >= 3.5:
                options.model_size = "small.en" if options.language == "en" else "small"
            elif vram_gb >= 2.5:
                options.model_size = "base.en" if options.language == "en" else "base"
            else:
                options.model_size = "tiny.en" if options.language == "en" else "tiny"
        else:
            options.model_size = "tiny"  # Fallback if no GPU is available
    else:
        ...
    _logger.info(
        f"Selected model size: {options.model_size} for language: {options.language}"
    )
    return options.model_size
