import sys
import os
import argparse
import logging

from utils.exceptions import FolderNotFoundError
from utils.constants import MODELS_AVAILABLE, WHISPER_LANGUAGE


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
        num_threads (str): Number of threads for transcription
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
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Options, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not Options._initialized:
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
            self._args = self._parser.parse_args()
            self.file = self._args.file
            self.directory = self._args.directory
            self.compute_device = self._args.compute_device
            self.model_size = self._args.model_size
            self.log_level = self._args.log_level
            self.language = self._args.language
            self.num_threads = self._args.num_threads
            self.txt = self._args.txt
            Options._initialized = True

            # If no args are passed to argparser, print help and exit
            if len(sys.argv) == 1:
                self._parser.print_help(sys.stdout)
                return

            # Set logging level
            logger = logging.getLogger("options")
            logger.setLevel(self.log_level)

            # If log level is less than INFO, set the progress bar to True
            self.print_progress_flag = (
                True
                if logging.getLevelName(self.log_level) < logging.getLevelName("INFO")
                else False
            )

            # Check that args.directory is a valid directory only if specified in the arguments
            if self.directory and not os.path.isdir(self.directory):
                logger.error(f"Error: Directory '{self.directory}' does not exist.")
                raise FolderNotFoundError
            # Check that args.file is a valid file only if specified in the arguments

            if self.file and not os.path.isfile(self.file):
                logger.error(f"Error: File '{self.file}' does not exist.")
                raise FileNotFoundError

            # Check that the language flag passed is compatible with Whisper
            if self.language and self.language not in WHISPER_LANGUAGE:
                logger.error(
                    f"The language code {self.language} is not a valid ISO 639-1 code supported by Whisper"
                )
                raise KeyError

    def __str__(self):
        """
        Return a string representation of the Options object.

        Returns:
            str: A string representation of the Options object containing its properties
                 and their current values.
        """
        args_str: str = f"""
        File: {self.file}
        Directory: {self.directory}
        Compute Device: {self.compute_device}
        Model Selected: {self.model_size}
        Log Level: {self.log_level}
        Media Language: {self.language}
        Number of Threads: {self.num_threads}
        Input Media Paths File: {self.txt}
        """
        return args_str

    def get_device(self):
        """
        Determine the best available device with graceful fallback to CPU.
        Returns:
            str: The device to use for computation.
        """
        from torch import cuda

        logger = logging.getLogger("get_device")

        if self.device_selection is None or "cuda" in self.device_selection.lower():
            try:
                if cuda.is_available():
                    logger.info("CUDA available. Using GPU acceleration.")
                    return "cuda"
                else:
                    logger.warning("CUDA not available, falling back to CPU")
            except Exception as e:
                logger.error(f"Warning: Error checking CUDA availability ({str(e)})")
                logger.warning("Falling back to CPU.")
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

        logger = logging.getLogger("get_model")

        if self.model_size not in MODELS_AVAILABLE:
            logger.error(f"Model size '{self.model_size}' is not available.")
            raise ValueError(
                f"Model size '{self.model_size}' is not valid. Available models: {MODELS_AVAILABLE}"
            )
        if self.model_size is None:
            # Check to see how much VRAM is available on your GPU and select the model accordingly
            if cuda.is_available():
                vram_gb = round(
                    (cuda.get_device_properties(0).total_memory / 1.073742e9), 1
                )
                logger.debug(f"Detected VRAM: {vram_gb} GB")
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
        logger.info(
            f"Selected model size: {self.model_size} for language: {self.language}"
        )
        return self.model_size

    # Ensure its a singleton
    @classmethod
    def get_instance(cls):
        """
        Get the singleton instance of the Options object.
        Returns:
            Options: The singleton instance of the Options object.
        """
        if cls._instance is None:
            cls._instance = Options()
        return cls._instance
