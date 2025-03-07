import sys
import os

from objects.options import Options
from objects.media import Media
from utils.audio_extractor import AudioExtractor


def _main():
    # Parse arguments
    options = Options()
    # Get list of valid media files
    media = Media(options)
    media_files = media.discover_media_files()

    # Extract audio
    extractor = AudioExtractor(media_files)
    try:
        audio_files = extractor.extract_audio_concurrent()
        # Use audio_files for transcription
    finally:
        extractor.cleanup()


if __name__ == "__main__":
    _main()
