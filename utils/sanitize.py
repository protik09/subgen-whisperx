import sys
import os
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

def sanitize_path(path: Optional[str]) -> Optional[str]:
    """
    Sanitize and validate a file or directory path.

    Args:
        path: The path to sanitize

    Returns:
        The sanitized path

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
    if re.match(r"^[a-z]+://", path.lower()):
        raise InvalidPathError("URL-like paths are not allowed")

    # Check for Windows reserved names directly on the input path
    if sys.platform == "win32":
        reserved_names = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "COM1",
            "COM2",
            "COM3",
            "COM4",
            "COM5",
            "COM6",
            "COM7",
            "COM8",
            "COM9",
            "LPT1",
            "LPT2",
            "LPT3",
            "LPT4",
            "LPT5",
            "LPT6",
            "LPT7",
            "LPT8",
            "LPT9",
        }

        # Check if path is exactly a reserved name or starts with a reserved name followed by dot
        path_parts = path.upper().split("\\")
        for part in path_parts:
            base_name = part.split(".")[0] if "." in part else part
            if base_name in reserved_names:
                raise InvalidPathError(f"'{path}' contains a reserved name on Windows")

    # Check for invalid characters in the path
    invalid_chars = '<>:"|?*'  # Removed backslash from this list
    if any(c in path for c in invalid_chars):
        raise InvalidPathError("Path contains invalid characters")

    # Additional validation for pipes and other special characters
    # These can be problematic on some systems
    if "|" in path or any(c in path for c in "[]{}"):
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
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "COM1",
            "COM2",
            "COM3",
            "COM4",
            "COM5",
            "COM6",
            "COM7",
            "COM8",
            "COM9",
            "LPT1",
            "LPT2",
            "LPT3",
            "LPT4",
            "LPT5",
            "LPT6",
            "LPT7",
            "LPT8",
            "LPT9",
        }

        for part in parts:
            # Remove extension for comparison
            base_name = part.split(".")[0]
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

def validate_thread_count(thread_count: Optional[str]) -> Optional[int]:
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
        raise InvalidThreadCountError(
            "Thread count contains command injection characters"
        )

    # Reject any thread counts with non-numeric characters
    cleaned_count = thread_count.strip()
    if not cleaned_count.isdigit() and not (
        cleaned_count.startswith("-") and cleaned_count[1:].isdigit()
    ):
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

def sanitize_language(language: Optional[str]) -> Optional[str]:
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
    if not language.strip().isalnum() and "-" not in language:
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