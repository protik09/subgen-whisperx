import sys
import os
import argparse
import logging
import re
from pathlib import Path
from typing import Optional

from utils.exceptions import (
    FolderNotFoundError,
    InvalidThreadCountError,
    InvalidPathError,
)
from utils.constants import MODELS_AVAILABLE, WHISPER_LANGUAGE

# Maximum allowed path length (Windows has 260 char limit by default)
MAX_PATH_LENGTH = 260
# Maximum reasonable thread count
MAX_THREADS = 128
# Minimum reasonable thread count
MIN_THREADS = 1


class Options:
    """
    A singleton class that handles command-line arguments and configuration options for the subtitle generator.
    This class implements the Singleton pattern to ensure only one instance of options is created
    throughout the application. It handles command-line argument parsing, validation, and provides
    methods to determine computation device and model selection based on available resources.
    Attributes:
        file (str): Path to the input media file
        directory (str): Path to directory containing media files
        compute_device (str): Device to use for computation ('cuda' or 'cpu')
        model_size (str): Whisper model size for transcription
        log_level (str): Logging level for the application
        language (str): Language code for subtitles
        num_threads (int): Number of threads for transcription
        txt (str): Path to text file containing media file paths
        print_progress_flag (bool): Flag to determine if progress bar should be shown
    Methods:
        get_device ( ): Determines the best available computation device
        get_model ( ): Selects appropriate Whisper model based on available resources
        get_instance ( ): Class method to get the singleton instance
    Raises:
        FolderNotFoundError: If specified directory does not exist
        FileNotFoundError: If specified file does not exist
        KeyError: If specified language is not supported by Whisper
        ValueError: If specified model size is not valid
        InvalidThreadCountError: If specified thread count is invalid
        InvalidPathError: If specified path is invalid
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Options, cls).__new__(cls)
        return cls._instance

    def _sanitize_path(self, path: Optional[str]) -> Optional[str]:
        """
        Sanitize and validate a file or directory path.

        Args:
            path (Optional[str]): The path to sanitize

        Returns:
            Optional[str]: The sanitized path

        Raises:
            InvalidPathError: If the path is invalid
        """
        if path is None:
            return None

        # Handle empty strings and whitespace
        if not path or not path.strip():
            raise InvalidPathError("Path cannot be empty")

        # Check for null bytes
        if "\0" in path:
            raise InvalidPathError("Path contains null bytes")

        # Normalize whitespace
        path = path.strip()

        # Check for control characters (including \t\n\r)
        if any(ord(c) < 32 for c in path):
            raise InvalidPathError("Path contains control characters")

        # Check for command injection attempts (more comprehensive pattern)
        if re.search(r"[;&|]", path):
            raise InvalidPathError("Path contains command injection characters")

        # Check for URL-like paths
        if re.match(r'^[a-z]+://', path.lower()):
            raise InvalidPathError("URL-like paths are not allowed")

        # Check for Windows reserved names directly on the input path
        if sys.platform == "win32":
            reserved_names = {
                "CON", "PRN", "AUX", "NUL",
                "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
                "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
            }
            
            # Check if path is exactly a reserved name or starts with a reserved name followed by dot
            path_parts = path.upper().split('\\')
            for part in path_parts:
                base_name = part.split('.')[0] if '.' in part else part
                if base_name in reserved_names:
                    raise InvalidPathError(f"'{path}' contains a reserved name on Windows")

        # Check for invalid characters in the path
        invalid_chars = '<>:"|?*'  # Removed backslash from this list
        if any(c in path for c in invalid_chars):
            raise InvalidPathError("Path contains invalid characters")

        # Additional validation for pipes and other special characters
        # These can be problematic on some systems
        if '|' in path or any(c in path for c in '[]{}'):
            raise InvalidPathError("Path contains special characters that may not be valid")

        # Check for excessive parent directory traversal
        if "../" * 10 in path or ".." * 20 in path:
            raise InvalidPathError("Excessive directory traversal")

        # Try to create a Path object to validate format
        try:
            path_obj = Path(path)
        except Exception:
            raise InvalidPathError("Invalid path format")

        # Check path length
        if len(str(path_obj)) > MAX_PATH_LENGTH:
            raise InvalidPathError(
                f"Path exceeds maximum length of {MAX_PATH_LENGTH} characters"
            )

        # Check for reserved names on Windows (case insensitive) for each part of the path
        if sys.platform == "win32" and len(path_obj.parts) > 0:
            # Check each part of the path for reserved names
            parts = [p.upper() for p in path_obj.parts]
            reserved_names = {
                "CON", "PRN", "AUX", "NUL",
                "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
                "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
            }
            
            for part in parts:
                # Remove extension for comparison
                base_name = part.split('.')[0]
                if base_name in reserved_names:
                    raise InvalidPathError(f"'{part}' contains a reserved name on Windows")

        # For absolute paths, resolve and convert to string
        if path_obj.is_absolute():
            try:
                path = str(path_obj.resolve(strict=False))
            except Exception:
                raise InvalidPathError("Unable to resolve absolute path")
        else:
            # For relative paths, normalize but preserve relativeness
            try:
                path = os.path.normpath(str(path_obj))
            except Exception:
                raise InvalidPathError("Unable to normalize path")

        return path

    def _validate_thread_count(self, thread_count: Optional[str]) -> Optional[int]:
        """
        Validate and convert thread count to integer.

        Args:
            thread_count (Optional[str]): The thread count to validate

        Returns:
            Optional[int]: The validated thread count

        Raises:
            InvalidThreadCountError: If the thread count is invalid
        """
        if thread_count is None:
            return None

        # Handle empty strings and whitespace
        if not thread_count or not thread_count.strip():
            raise InvalidThreadCountError("Thread count cannot be empty")

        # Check for newlines and control characters first - these can be used for command injection
        if any(ord(c) < 32 for c in thread_count):
            raise InvalidThreadCountError("Thread count contains control characters")

        # Check for command injection attempts
        if re.search(r"[;&|]", thread_count):
            raise InvalidThreadCountError("Thread count contains command injection characters")

        # Reject any thread counts with non-numeric characters
        cleaned_count = thread_count.strip()
        if not cleaned_count.isdigit() and not (cleaned_count.startswith('-') and cleaned_count[1:].isdigit()):
            raise InvalidThreadCountError("Thread count must contain only digits")

        # Remove any potential injection attempts and whitespace
        parts = [p for p in thread_count.split() if p.strip()]
        if not parts:
            raise InvalidThreadCountError("Invalid thread count format")
        
        # Only take the first part to avoid command injection
        thread_count = parts[0].strip()

        # Check length to prevent overflow
        if len(thread_count) > 10:  # Being generous here
            raise InvalidThreadCountError("Thread count value is too long")

        # Try to convert to float first to catch scientific notation and decimals
        try:
            float_val = float(thread_count)
        except ValueError:
            raise InvalidThreadCountError("Thread count must be a valid number")

        # Check if it's actually an integer
        if float_val != int(float_val):
            raise InvalidThreadCountError("Thread count must be an integer")

        count = int(float_val)

        # Validate range
        if count < MIN_THREADS or count > MAX_THREADS:
            raise InvalidThreadCountError(
                f"Thread count must be between {MIN_THREADS} and {MAX_THREADS}"
            )
        
        return count

    def _sanitize_model_size(self, model_size: Optional[str]) -> Optional[str]:
        """
        Sanitize and validate model size.

        Args:
            model_size (Optional[str]): The model size to validate

        Returns:
            Optional[str]: The validated model size

        Raises:
            ValueError: If the model size is invalid
        """
        if model_size is None:
            return None

        # Handle empty strings and whitespace
        if not model_size.strip():
            raise ValueError("Model size cannot be empty")

        # Remove any potential injection attempts and whitespace
        try:
            model_size = model_size.split()[0].strip()
        except IndexError:
            raise ValueError("Invalid model size format")

        # Check for invalid characters
        if any(c in model_size for c in '\0\n\r\t'):
            raise ValueError("Model size contains invalid characters")

        # Check length
        if len(model_size) > 20:  # Being generous here
            raise ValueError("Model size is too long")

        if model_size not in MODELS_AVAILABLE:
            raise ValueError(
                f"Model size '{model_size}' is not valid. Available models: {MODELS_AVAILABLE}"
            )
        return model_size

    def _sanitize_language(self, language: Optional[str]) -> Optional[str]:
        """
        Sanitize and validate language code.

        Args:
            language (Optional[str]): The language code to validate

        Returns:
            Optional[str]: The validated language code

        Raises:
            KeyError: If the language code is invalid
        """
        if language is None:
            return None

        # Handle empty strings and whitespace
        if not language or not language.strip():
            raise KeyError("Language code cannot be empty")

        # Check for newlines and control characters first - these can be used for command injection
        if any(ord(c) < 32 for c in language):
            raise KeyError("Language code contains control characters")

        # Check for command injection attempts
        if re.search(r"[;&|]", language):
            raise KeyError("Language code contains command injection characters")

        # Reject any language codes with spaces or non-alphanumeric characters
        if not language.strip().isalnum() and '-' not in language:
            raise KeyError("Language code contains invalid characters")

        # Split on any whitespace and take first part
        parts = [p for p in language.split() if p.strip()]
        if not parts:
            raise KeyError("Invalid language code format")
        
        # Only take the first part to avoid command injection
        language = parts[0].strip()

        # Check length (ISO 639-1 codes are 2-3 chars)
        if len(language) > 10:  # Being generous here
            raise KeyError("Language code is too long")

        # Validate against known languages
        if language not in WHISPER_LANGUAGE:
            raise KeyError(
                f"The language code {language} is not a valid ISO 639-1 code supported by Whisper"
            )
        
        return language

    def __init__(self):
        if not Options._initialized:
            # Pre-validate paths before argparse to catch malformed paths
            raw_args = sys.argv[1:]
            path_options = ['-f', '--file', '-d', '--directory', '-t', '--txt']
            
            for i in range(len(raw_args) - 1):
                if raw_args[i] in path_options:
                    # Don't catch the exception, let it propagate
                    self._sanitize_path(raw_args[i + 1])
                elif raw_args[i] in ['-l', '--language']:
                    # Don't catch the exception, let it propagate
                    self._sanitize_language(raw_args[i + 1])
                elif raw_args[i] in ['-n', '--num_threads']:
                    # Don't catch the exception, let it propagate
                    self._validate_thread_count(raw_args[i + 1])

            self._parser = argparse.ArgumentParser(description="Subtitle Generator")
            self._parser.add_argument(
                "-f",
                "--file",
                default=None,
                help="Path to the input media file",
            )
            self._parser.add_argument(
                "-d",
                "--directory",
                default=None,
                help="Path to directory containing media files",
            )
            self._parser.add_argument(
                "-c",
                "--compute_device",
                default=None,
                choices=["cuda", "cpu"],
                help="Device to use for computation (cuda or cpu)",
            )
            self._parser.add_argument(
                "-m",
                "--model_size",
                default=None,
                choices=MODELS_AVAILABLE,
                help="Whisper model size to use for transcription (default: auto-select based on VRAM)",
            )
            self._parser.add_argument(
                "-log",
                "--log_level",
                default="INFO",
                choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                help="Set the logging level (default: ERROR)",
            )
            self._parser.add_argument(
                "-l",
                "--language",
                default=None,
                help="Set the language for subtitles",
            )
            self._parser.add_argument(
                "-n",
                "--num_threads",
                default=None,
                help="Set the number of threads for transcription",
            )
            self._parser.add_argument(
                "-t",
                "--txt",
                default=None,
                help="Pass a txt file containing the paths to either media files or directories",
            )

            try:
                self._args = self._parser.parse_args()
            except Exception as e:
                _logger = logging.getLogger("options")
                _logger.error(f"Error parsing arguments: {str(e)}")
                raise

            # Set up logging first
            self.log_level = self._args.log_level
            _logger = logging.getLogger("options")
            _logger.setLevel(self.log_level)

            # Sanitize and validate all inputs
            # Remove the try-except block and just call the validation functions directly
            self.file = self._sanitize_path(self._args.file)
            self.directory = self._sanitize_path(self._args.directory)
            self.txt = self._sanitize_path(self._args.txt)
            self.compute_device = self._args.compute_device
            self.model_size = self._sanitize_model_size(self._args.model_size)
            self.language = self._sanitize_language(self._args.language)
            self.num_threads = self._validate_thread_count(self._args.num_threads)

            Options._initialized = True

            # If no args are passed to argparser, print help and exit
            if len(sys.argv) == 1:
                self._parser.print_help(sys.stdout)
                return

            # If log level is less than INFO, set the progress bar to True
            self.print_progress_flag = (
                True
                if logging.getLevelName(self.log_level) < logging.getLevelName("INFO")
                else False
            )

            # Validate existence of paths after sanitization
            if self.directory and not os.path.isdir(self.directory):
                _logger.error(f"Error: Directory '{self.directory}' does not exist.")
                raise FolderNotFoundError(
                    message=f"Directory '{self.directory}' does not exist."
                )

            if self.file and not os.path.isfile(self.file):
                _logger.error(f"Error: File '{self.file}' does not exist.")
                raise FileNotFoundError(f"File '{self.file}' does not exist.")

            if self.txt and not os.path.isfile(self.txt):
                _logger.error(f"Error: Text file '{self.txt}' does not exist.")
                raise FileNotFoundError(f"Text file '{self.txt}' does not exist.")

    def __str__(self):
        """
        Return a string representation of the Options object.

        Returns:
            str: A string representation of the Options object containing its properties
                 and their current values.
        """
        # Normalize paths for consistent representation
        file_path = os.path.normpath(str(self.file)) if self.file else None
        dir_path = os.path.normpath(str(self.directory)) if self.directory else None
        txt_path = os.path.normpath(str(self.txt)) if self.txt else None

        args_str: str = f"""
        File: {file_path}
        Directory: {dir_path}
        Compute Device: {self.compute_device}
        Model Selected: {self.model_size}
        Log Level: {self.log_level}
        Media Language: {self.language}
        Number of Threads: {self.num_threads}
        Input Media Paths File: {txt_path}
        """
        return args_str

    def get_device(self):
        """
        Determine the best available device with graceful fallback to CPU.
        Returns:
            str: The device to use for computation.
        """
        from torch import cuda

        _logger = logging.getLogger("options_get_device")

        if self.compute_device is None or "cuda" in self.compute_device.lower():
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

    def get_model(self):
        """
        Get the Whisper model size to use for transcription.
        Returns:
            str: The Whisper model size to use for transcription.
        """
        from torch import cuda

        _logger = logging.getLogger("options_get_model")

        if self.model_size not in MODELS_AVAILABLE:
            _logger.error(f"Model size '{self.model_size}' is not available.")
            raise ValueError(
                f"Model size '{self.model_size}' is not valid. Available models: {MODELS_AVAILABLE}"
            )
        if self.model_size is None:
            # Check to see how much VRAM is available on your GPU and select the model accordingly
            if cuda.is_available():
                vram_gb = round(
                    (cuda.get_device_properties(0).total_memory / 1.073742e9), 1
                )
                _logger.debug(f"Detected VRAM: {vram_gb} GB")
                if vram_gb >= 9.0:
                    self.model_size = "large-v2"
                elif vram_gb >= 7.5:
                    self.model_size = "medium"
                elif vram_gb >= 4.5:
                    self.model_size = "small.en" if self.language == "en" else "small"
                elif vram_gb >= 3.5:
                    self.model_size = "small.en" if self.language == "en" else "small"
                elif vram_gb >= 2.5:
                    self.model_size = "base.en" if self.language == "en" else "base"
                else:
                    self.model_size = "tiny.en" if self.language == "en" else "tiny"
            else:
                self.model_size = "tiny"  # Fallback if no GPU is available
        else:
            self.model_size = self.model_size
        _logger.info(
            f"Selected model size: {self.model_size} for language: {self.language}"
        )
        return self.model_size
