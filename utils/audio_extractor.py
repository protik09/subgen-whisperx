import logging
import concurrent.futures
from pathlib import Path
from typing import List
import tempfile

import ffmpeg

from objects.mediafile import MediaFile
from objects.audiofile import AudioFile


class AudioExtractor:
    """Handles extraction of audio from media files."""

    def __init__(self, media_files: List[MediaFile]) -> None:
        self._logger = logging.getLogger(__name__)
        self._temp_dir = Path(tempfile.gettempdir()) / "subgen_audio"
        self._temp_dir.mkdir(exist_ok=True)
        self._media_files = media_files

    def extract_audio_concurrent(self) -> List[AudioFile]:
        """
        Extract audio from multiple media files concurrently.

        Args:
            media_files (List[MediaFile]): List of valid media files

        Returns:
            List[AudioFile]: List of extracted audio files
        """
        self._logger.info(f"Extracting audio from {len(self._media_files)} media files...")
        audio_files: List[AudioFile] = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_media = {
                executor.submit(self._extract_audio, media_file): media_file
                for media_file in self._media_files
            }

            for future in concurrent.futures.as_completed(future_to_media):
                media_file = future_to_media[future]
                try:
                    audio_file = future.result()
                    if audio_file:
                        audio_files.append(audio_file)
                except Exception as e:
                    self._logger.error(
                        f"Failed to extract audio from {media_file.path}: {str(e)}"
                    )

        self._logger.info(f"Successfully extracted {len(audio_files)} audio files")
        return audio_files

    def _extract_audio(self, media_file: MediaFile) -> AudioFile:
        """
        Extract audio from a single media file.

        Args:
            media_file (MediaFile): Media file to extract audio from

        Returns:
            AudioFile: Extracted audio file object
        """
        if media_file.is_audio_only:
            # If the file is already audio-only, no need to extract
            return AudioFile(
                path=media_file.path,
                source_media_path=media_file.path,
                is_original_audio=True,
            )

        # Generate output path in temp directory
        output_path = self._temp_dir / f"{media_file.path.stem}_audio.wav"

        try:
            # Extract audio using ffmpeg
            (
                ffmpeg.input(str(media_file.path))
                .output(
                    str(output_path),
                    acodec="pcm_s16le",  # Use WAV format for compatibility
                    ac=1,  # Mono audio
                    ar="16k",  # 16kHz sample rate
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

            self._logger.debug(f"Successfully extracted audio to {output_path}")
            return AudioFile(
                path=output_path,
                source_media_path=media_file.path,
                is_original_audio=False,
            )

        except ffmpeg.Error as e:
            self._logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise
        except Exception as e:
            self._logger.error(f"Unexpected error: {str(e)}")
            raise

    def cleanup(self):
        """Clean up temporary files."""
        try:
            for file in self._temp_dir.glob("*"):
                if not file.is_file():
                    continue
                try:
                    file.unlink()
                except Exception as e:
                    self._logger.warning(f"Failed to delete {file}: {e}")
            self._temp_dir.rmdir()
        except Exception as e:
            self._logger.warning(f"Failed to cleanup temp directory: {e}")
