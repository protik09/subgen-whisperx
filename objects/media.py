import os
import logging
import concurrent.futures
from typing import List, Set, Tuple
from pathlib import Path

import ffmpeg

from objects.args import Options
from objects.mediafile import MediaFile
from utils.exceptions import MediaNotFoundError
from utils.constants import MEDIA_EXTENSIONS


# Function to check if media file is valid
def is_media_file(file_path: str) -> Tuple[bool, bool]:
    """Check if a file is a valid media file.

    Args:
        file_path (str): Path to the file to check

    Returns:
        Tuple[bool, bool]: Tuple containing (is_valid_media, is_audio_only)
    """
    logger = logging.getLogger("subgen_whisperx.is_media_file")
    _valid_media_flag: bool = False
    _valid_audio_flag: bool = False
    try:
        # This weird thing exists because ffmpeg.probe() shows a text file as a valid video file
        probe = (
            ffmpeg.probe(file_path)
            if os.path.split(file_path)[1].split(".")[-1] != "txt"
            else None
        )
        # Ensure probe is not None before proceeding
        if probe and len(probe["streams"]) > 0:
            stream_type: str = probe["streams"][0]["codec_type"]
            if stream_type == "audio" or stream_type == "video":
                _valid_media_flag = True
                if stream_type == "audio":
                    _valid_audio_flag = True
        logger.debug(
            f"File: {file_path}, Valid Media: {_valid_media_flag}, Audio Only: {_valid_audio_flag}"
        )
    except Exception as e:
        _valid_audio_flag = False
        _valid_media_flag = False
        logger.error(f"An error occurred while probing the file: {e}")
    finally:
        return _valid_media_flag, _valid_audio_flag

class Media:
    """
    Media class to handle the media files and its properties.
    """

    def __init__(self, options: Options):
        """
        Initialize the Media object with the Options object.

        Args:
            options (Options): An Options object containing the parsed arguments.
        """
        self._options = options
        self._logger = logging.getLogger(__name__)
        self._media_files: List[MediaFile] = []
        self._extracted_audio_paths: List[tuple[str, str, bool]] = []

    # @staticmethod
    def get_media_files(self) -> List[MediaFile]:
        """
        Discover media files based on the options provided.

        Returns:
            List[MediaFile]: List of valid MediaFile objects

        Raises:
            MediaNotFoundError: If no valid media files are found
        """
        self._logger.info("Discovering media files...")
        potential_media_files: Set[Path] = self._collect_potential_files()

        if not potential_media_files:
            self._logger.error("No potential media files found")
            raise MediaNotFoundError("No potential media files found")

        # Validate files concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_file = {
                executor.submit(self.is_media_file, file_path): file_path
                for file_path in potential_media_files
            }

            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    is_valid, is_audio = future.result()
                    if is_valid:
                        self._media_files.append(MediaFile(file_path, is_audio))
                    else:
                        self._logger.warning(
                            f"Skipping invalid media file: '{file_path}'"
                        )
                except Exception as e:
                    self._logger.error(f"Validation failed for {file_path}: {str(e)}")

        if not self._media_files:
            self._logger.error("No valid media files found")
            raise MediaNotFoundError("No valid media files found")

        self._logger.info(f"Discovered {len(self._media_files)} valid media files")
        return self._media_files

    def _collect_potential_files(self) -> Set[Path]:
        """
        Collect all potential media files from the provided sources.

        Returns:
            Set[Path]: Set of potential media file paths
        """
        potential_files: Set[Path] = set()

        # Handle single file
        if self._options.file and os.path.isfile(self._options.file):
            potential_files.add(Path(self._options.file))

        # Handle directory
        if self._options.directory and os.path.isdir(self._options.directory):
            for root, _, files in os.walk(self._options.directory):
                for f in files:
                    if f.lower().endswith(tuple(MEDIA_EXTENSIONS)):
                        potential_files.add(Path(root) / f)

        # Handle text file with paths
        if self._options.txt and os.path.isfile(self._options.txt):
            try:
                with open(self._options.txt, "r") as f:
                    for line in f:
                        path = Path(line.strip())
                        if path.is_file():
                            potential_files.add(path)
                        elif path.is_dir():
                            for root, _, files in os.walk(path):
                                for f in files:
                                    if f.lower().endswith(tuple(MEDIA_EXTENSIONS)):
                                        potential_files.add(Path(root) / f)
            except Exception as e:
                self._logger.error(f"Error reading text file {self._options.txt}: {e}")
                raise

        return potential_files

    def is_media_file(self, file_path: Path) -> tuple[bool, bool]:
        """
        Check if a file is a valid media file.

        Args:
            file_path (Path): Path to the file to check

        Returns:
            Tuple[bool, bool]: Tuple containing (is_valid_media, is_audio_only)
        """
        _valid_media_flag: bool = False
        _valid_audio_flag: bool = False
        try:
            # This check exists because ffmpeg.probe() might show a text file as a valid video file
            if file_path.suffix.lower() == ".txt":
                return False, False

            probe = ffmpeg.probe(str(file_path))
            # Ensure probe is not None before proceeding
            if probe and len(probe["streams"]) > 0:
                stream_type: str = probe["streams"][0]["codec_type"]
                if stream_type == "audio" or stream_type == "video":
                    _valid_media_flag = True
                    if stream_type == "audio":
                        _valid_audio_flag = True
            self._logger.debug(
                f"File: {file_path}, Valid Media: {_valid_media_flag}, Audio Only: {_valid_audio_flag}"
            )
            return _valid_media_flag, _valid_audio_flag
        except Exception as e:
            self._logger.error(f"An error occurred while probing the file: {e}")
            return False, False
