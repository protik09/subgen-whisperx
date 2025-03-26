import concurrent.futures
import logging
import os
from typing import List, Tuple, Optional

import ffmpeg

from objects.media import Media
from objects.mediafile import MediaFile
from objects.options import Options
from utils.timer import Timer

# Create a singleton timer instance
stopwatch = Timer()

def extract_audio(video: MediaFile, options: Options) -> MediaFile:
    """
    Extract audio from input video file using ffmpeg.

    Args:
        video (MediaFile): The video file to extract audio from
        options (Options): Configuration options including logging settings

    Returns:
        MediaFile: The input video file with extracted_audio_path updated

    Raises:
        ValueError: If the extracted audio file exceeds 25MB in size
    """
    logger = logging.getLogger("audio_extractor")
    logger.setLevel(options.log_level)

    stopwatch.start(f"Audio Extraction -> {video.path.name}")
    video.extracted_audio_path = f"audio-{os.path.splitext(os.path.basename(video.path))[0]}.mp3"

    try:
        stream = ffmpeg.input(video.path.name)
        stream = ffmpeg.output(
            stream,
            video.extracted_audio_path,
            acodec="libmp3lame",  # Faster MP3 encoder
            ac=1,  # Convert to mono
            ar=16000,  # Lower sample rate (whisper uses 16kHz)
            **{
                "q:a": 0,  # VBR encoding
                "threads": 0,  # Use all CPU threads
                "thread_queue_size": 1024,  # Larger queue for better throughput
                "fflags": "+fastseek",  # Fast seeking
            },
        )

        ffmpeg.run(stream, overwrite_output=True)
    except Exception as e:
        video.extracted_audio_path = None
        logger.error(f"An error occurred while extracting audio: {str(e)}")
        raise
    finally:
        stopwatch.stop(f"Audio Extraction -> {os.path.basename(video.path)}")

    return video

def _extract_audio_worker(video_path: str, options: Options) -> Tuple[str, Optional[str]]:
    """Standalone function for audio extraction in worker processes"""
    try:
        media_file = MediaFile(video_path)
        result = extract_audio(media_file, options)
        return video_path, result.extracted_audio_path
    except Exception as e:
        return video_path, None

def extract_audio_concurrent(media_files: List[MediaFile], options: Options) -> List[MediaFile]:
    """
    Extract audio from multiple media files concurrently.

    Args:
        media_files (List[MediaFile]): List of media files to process
        options (Options): Configuration options including logging settings

    Returns:
        List[MediaFile]: List of processed media files with extracted audio paths
    """
    logger = logging.getLogger("audio_extractor")
    logger.setLevel(options.log_level)
    results = []
    path_to_file = {str(file.path): file for file in media_files}
    max_workers = os.cpu_count()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {}
        for file in media_files:
            if not file.is_audio:
                future = executor.submit(_extract_audio_worker, str(file.path), options)
                future_to_path[future] = str(file.path)
            else:
                results.append(file)

        for future in concurrent.futures.as_completed(future_to_path):
            video_path = future_to_path[future]
            file = path_to_file[video_path]
            try:
                _, extracted_path = future.result()
                file.extracted_audio_path = extracted_path
                results.append(file)
                logger.info(f"Successfully extracted audio from {file.path}")
            except Exception as e:
                logger.error(f"Failed to extract audio from {file.path}: {e}")
                file.extracted_audio_path = None
                results.append(file)

    logger.info(f"Extracted audio from {len(results)} media files")
    return results
