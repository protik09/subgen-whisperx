import os
import pathlib
import json
import logging
import logging.config
from datetime import datetime

import coloredlogs

from objects.app_options import AppOptions, parse_arguments
from utils.sodll import setup_dll_paths
# from objects.media import Media
# from objects.mediafile import MediaFile
# from objects.audio_extractor import ConcurrentAudioExtractor

# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(
    log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_subgen.log"
)
LOGGING_LEVEL = logging.DEBUG

# Create the logger before configuration
logger = logging.getLogger("subgen_whisperx")
logger.setLevel(LOGGING_LEVEL)

# Load and apply logging configuration
config_file = pathlib.Path("log_format.json")
try:
    with open(config_file, "r") as f:
        log_config = json.load(f)
        log_config["handlers"]["file"]["filename"] = log_filename
        log_config["handlers"]["file"]["level"] = LOGGING_LEVEL
        logging.config.dictConfig(log_config)
except Exception:
    # Fallback to basic file handler if config fails
    file_handler = logging.FileHandler(filename=log_filename, mode="a")
    file_handler.setLevel(LOGGING_LEVEL)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - [%(name)s] - [%(levelname)s] : %(message)s")
    )
    logger.addHandler(file_handler)

# Install coloredlogs for console output
coloredlogs.install(logger=logger)


def main():
    # Init Logging for the program
    logger = logging.getLogger("main")
    logger.setLevel(options.log_level)
    coloredlogs.install(
        logger=logger,
        level=options.log_level,
        milliseconds=True,
        fmt=log_config["formatters"]["detailed"]["format"],
    )
    logger.debug(f"Options selected are -> {options}")
    

    # Get list of valid media files
    # media: Media = Media(options)
    # media_files: list[MediaFile] = media.get_media_files()

    # # Extract audio
    # extractor = ConcurrentAudioExtractor(options=options)
    # try:
    #     audio_files = extractor.extract_audio_concurrent(media_files=media_files)
    #     # Use audio_files for transcription
    # finally:
    #     # extractor.cleanup()
        # ...


if __name__ == "__main__":
    # Ensure DLL's are present in the global namespace
    setup_dll_paths()
    try:
    # Parse arguments, returns Options object for CLI or None for GUI
    options: AppOptions = parse_arguments()
    
    
    # Check if running in CLI or GUI mode
    main()
