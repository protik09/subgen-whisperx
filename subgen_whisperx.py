import sys
import os
import ffmpeg
import utils.timer as timer
import logging
import coloredlogs
from datetime import datetime
from typing import Union, Dict, List, Tuple
from utils.constants import DEFAULT_INPUT_VIDEO, MODELS_AVAILABLE
import argparse
import srt
# from halo import Halo

# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(
    log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_subgen.log"
)
LOGGING_LEVEL = logging.ERROR
logging.basicConfig(filename=log_filename, filemode="a", level=LOGGING_LEVEL)
coloredlogs.install(level="DEBUG")

# Init global timer
stopwatch: timer.Timer = timer.Timer("DEBUG")


def get_device(device_selection: str | None = None) -> str:
    """Determine the best available device with graceful fallback"""
    from torch import cuda

    logger = logging.getLogger("get_device")

    if device_selection is None or "cuda" in device_selection.lower():
        try:
            if cuda.is_available():
                logger.debug("CUDA available.")
                return "cuda"
            else:
                logger.warning("CUDA not available, falling back to CPU")
        except Exception as e:
            logger.error(f"Warning: Error checking CUDA availability ({str(e)})")
            logger.warning("Falling back to CPU.")
    else:
        pass

    return "cpu"


def get_model(model_size: str | None = None, language: str | None = None) -> str:
    """Select the model based on size and language."""
    from torch import cuda

    logger = logging.getLogger("get_model")

    if model_size not in MODELS_AVAILABLE:
        logger.error(f"Model size '{model_size}' is not available.")
        raise ValueError(
            f"Model size '{model_size}' is not valid. Available models: {MODELS_AVAILABLE}"
        )
    if model_size is None:
        # Check to see how much VRAM is available on your GPU and select the model accordingly
        if cuda.is_available():
            vram_gb = round(
                (cuda.get_device_properties(0).total_memory / 1.073742e9), 1
            )
            logger.debug(f"Detected VRAM: {vram_gb} GB")
            if vram_gb >= 9.0:
                model_size = "large-v2"
            elif vram_gb >= 7.5:
                model_size = "medium"
            elif vram_gb >= 4.5:
                model_size = "small.en" if language == "en" else "small"
            elif vram_gb >= 3.5:
                model_size = "small.en" if language == "en" else "small"
            elif vram_gb >= 2.5:
                model_size = "base.en" if language == "en" else "base"
            else:
                model_size = "tiny.en" if language == "en" else "tiny"
        else:
            model_size = "tiny"  # Fallback if no GPU is available
    else:
        model_size = model_size

    logger.info(f"Selected model size: {model_size} for language: {language}")
    return model_size


# Function to check if media file is valid
def is_media_file(file_path: str) -> Tuple[bool, bool]:
    """Check if a file is a valid media file.

    Args:
        file_path (str): Path to the file to check

    Returns:
        Tuple[bool, bool]: Tuple containing (is_valid_media, is_audio_only)
    """
    logger = logging.getLogger("is_media_file")
    _valid_media_flag: bool = False
    _valid_audio_flag: bool = False
    try:
        probe = ffmpeg.probe(file_path)
        # Check whether a media stream exists in the file
        if len(probe["streams"]) > 0:
            stream_type: str = probe["streams"][0]["codec_type"]
            if stream_type == "audio" or stream_type == "video":
                _valid_media_flag = True
                if stream_type == "audio":
                    _valid_audio_flag = True
        return _valid_media_flag, _valid_audio_flag
    except Exception as e:
        logger.error(f"An error occurred while probing the file: {e}")
        return False, False


def get_media_files(directory: str = None, file: str = None) -> List[Tuple[str, bool]]:
    """Get list of valid media files from directory and/or single file.

    Args:
        directory (str, optional): Directory path to search for media files
        file (str, optional): Single file path to check

    Returns:
        List[Tuple[str, bool]]: List of tuples containing (file_path, is_audio_flag)
    """
    logger = logging.getLogger("get_media_files")
    media_files: List[Tuple[str, bool]] = []

    if directory:
        for root, _, files in os.walk(directory):
            for f in files:
                file_path = os.path.join(root, f)
                is_valid, is_audio = is_media_file(file_path)
                if is_valid:
                    media_files.append((file_path, is_audio))

        if not media_files:
            logger.error(f"No valid media files found in directory '{directory}'")
            return None

    if file:
        is_valid, is_audio = is_media_file(file)
        if is_valid:
            media_files.append((file, is_audio))
        else:
            logger.error(f"Error: File '{file}' is not a valid media file.")
            return None

    return media_files


def extract_audio(video_path: str = DEFAULT_INPUT_VIDEO) -> str:
    """Extract audio from input video file using ffmpeg.
    This function extracts the audio track from the input video file and converts it to MP3 format
    with optimized settings for transcription (mono, 16kHz sample rate). The extraction process
    uses ffmpeg with performance optimizations like multi-threading and VBR encoding.
    Returns:
        str: Path to the extracted audio file (format: 'audio-{INPUT_VIDEO_NAME}.mp3')
    Raises:
        ValueError: If the extracted audio file exceeds 25MB in size
    Notes:
        - Uses libmp3lame codec for faster MP3 encoding
        - Converts audio to mono channel
        - Downsamples to 16kHz for compatibility with Whisper
        - Uses variable bitrate (VBR) encoding
        - Utilizes all available CPU threads for processing
        - Uses a larger thread queue size for better throughput
        - Enables fast seeking for improved performance
    """
    logger = logging.getLogger("extract_audio")
    stopwatch.start("Audio Extraction")
    extracted_audio_path: str = (
        f"audio-{os.path.splitext(os.path.basename(video_path))[0]}.mp3"
    )

    try:
        # Add optimization flags to ffmpeg
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(
            stream,
            extracted_audio_path,
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
        logger.error(f"An error occurred while extracting audio: {e}")
    stopwatch.stop("Audio Extraction")
    return extracted_audio_path


# @Halo(text="Transcribing....", text_color="green", spinner="dots", placement="right")
def get_transcription(
    audio_path: str,
    device: str,
    model_size: str,
    print_progress: bool = False,
    language: str = None,
    num_threads: int = None,
) -> Tuple[str, List[Dict[str, Union[float, str]]]]:
    """
    Transcribes audio file using WhisperX model and aligns timestamps.
    It handles model loading, transcription, and alignment in one workflow.
    Args:
        audio_path (str): Path to the audio file to transcribe
        device (str): Device to run inference on ('cuda' or 'cpu')
        model_size (str): Size of the Whisper model to use (e.g. 'tiny', 'base', 'small', 'medium', 'large')
        print_progress (bool, optional): Whether to print progress during transcription. Defaults to False.
        language (str, optional): Language code for transcription. If None, auto-detects language. Defaults to None.
        num_threads (int, optional): Number of CPU threads to use. If None, uses all available threads. Defaults to None.
    Returns:
        Tuple[str, List[Dict[str, Union[float, str]]]]: A tuple containing:
            - language (str): Detected or specified language code
            - segments (list): List of transcribed segments with aligned timestamps.
                              Each segment is a dict containing:
                              - 'start': Start time in seconds
                              - 'end': End time in seconds
                              - 'text': Transcribed text
    Notes:
        - Uses int8 quantization for compute optimization
        - Automatically detects language if not specified
    """
    import whisperx

    logger = logging.getLogger("transcribe")
    stopwatch.start("Transcription")

    # Set number of threads for transcription
    threads_available: int = os.cpu_count()
    if num_threads is None or num_threads < 1 or num_threads > threads_available:
        num_threads = threads_available

    # Load model
    model = whisperx.load_model(
        whisper_arch=model_size,
        device=device,
        compute_type="int8",
        language=language,
        threads=num_threads,
    )

    # Initial transcription
    initial_result: Dict = model.transcribe(
        audio=audio_path,
        batch_size=16,
        print_progress=print_progress,
    )

    # Store language before alignment
    if language is None:
        language = initial_result["language"]

    # Align timestamps
    model_a, metadata = whisperx.load_align_model(
        language_code=language,
        device=device,
    )
    aligned_result: Dict = whisperx.align(
        transcript=initial_result["segments"],
        model=model_a,
        align_model_metadata=metadata,
        audio=audio_path,
        device=device,
        print_progress=print_progress,
    )

    # Get aligned segments
    segments: List[Dict[str, Union[float, str]]] = aligned_result["segments"]

    logger.info(f"Language: {language}")
    for segment in segments:
        logger.debug(
            f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}"
        )

    # Delete audio file only if orginal file was a video after transcription
    # try:
    #     os.remove(audio_path)
    # except Exception as e:
    #     logger.error(f"An error occurred while deleting audio file: {e}")

    stopwatch.stop("Transcription")
    return language, segments


def generate_subtitles(segments: Dict) -> str:
    logger = logging.getLogger("generate_subtitles")
    _srt_content = []
    for i, segment in enumerate(segments, start=1):
        segment_start = timer.Timer.format_time(segment["start"])
        segment_end = timer.Timer.format_time(segment["end"])
        text = segment["text"].strip()

        # SRT format: [segment number] [start] --> [end] [text]
        _srt_content.append(f"{i}")
        _srt_content.append(f"{segment_start} --> {segment_end}")
        _srt_content.append(f"{text}{os.linesep}")

    return f"{os.linesep}".join(_srt_content)


def post_process(subtitles: str) -> str:
    logger = logging.getLogger("post_process")
    """Post-process the generated subtitles.
    This function performs additional processing on the generated subtitles to improve readability
    and ensure compliance with common subtitle standards.
    Args:
        subtitles (list): The generated subtitles as a list of strings
    Returns:
        str: The post-processed subtitles as a single string
    """

    # Clip lines that go over 150 characters taking into account word boundaries
    _subtitles_clean: str = ""
    for line in subtitles:
        if len(line) > 150:
            try:
                line = line[:150].rsplit(" ", 1)[0]
            except ValueError:
                logger.warning(
                    f"Line too long and cannot be split: {line}. Clipping to 150 characters."
                )
                line = line[:150]
        else:
            pass

        _subtitles_clean += line

    # Make legal SRT from the generated subtitles
    try:
        _subtitles_clean = srt.make_legal_content(_subtitles_clean)
    except Exception as e:
        logger.error(
            f"An error occurred while parsing SRT. No subtitles will be written to file: {e}"
        )
        _subtitles_clean = ""

    return _subtitles_clean


def write_subtitles(
    subtitles: str, file_name: str, input_media_path: str, language: str
) -> None:
    """Write the generated subtitles to a file.
    Args:
        subtitles (str): The generated subtitles as a single string
        output_path (str): The path to write the subtitles to
        language (str): The language for subtitles
    """
    logger = logging.getLogger("write_subtitles")
    # The following should generate something like "input.ai.srt" from "input.mp4"
    _subtitle_file_name = f"{file_name}.ai-{language}.srt"

    # Write subtitles to file
    subtitle_path = os.path.join(os.path.dirname(input_media_path), _subtitle_file_name)
    try:
        with open(subtitle_path, "w", encoding="utf-8") as f:
            f.write(subtitles)
            logger.info(f"Subtitle file generated: {_subtitle_file_name}")
    except Exception as e:
        logger.error(f"An error occurred while writing the subtitle file: {e}")


def main():
    logger = logging.getLogger("main")
    # Parse command line arguments
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
        help="Device to use for computation (cuda or cpu)",
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
        default="ERROR",
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
    args = parser.parse_args()

    # Set logging level
    logging_level = getattr(logging, args.log_level.upper(), logging.DEBUG)
    logging.getLogger().setLevel(logging_level)
    coloredlogs.install(level=logging_level)
    # Set print_prgress flag depending on logging level
    print_progress = logging_level <= logging.INFO

    # If no args are passed to argparser, print help and exit
    if len(sys.argv) == 1:
        parser.print_help(sys.stdout)
        sys.exit()

    # Check that args.directory is a valid directory only if specified in the arguments
    if args.directory and not os.path.isdir(args.directory):
        logger.error(f"Error: Directory '{args.directory}' does not exist.")
        return
    # Check that args.file is a valid file only if specified in the arguments
    if args.file and not os.path.isfile(args.file):
        logger.error(f"Error: File '{args.file}' does not exist.")
        return

    media_files = get_media_files(args.directory, args.file)
    if not media_files:
        return

    # Process each media file
    for media_file in media_files:
        input_media_path = media_file[0]
        audio_flag = media_file[1]
        file_name = str(os.path.basename(input_media_path.rsplit(".", 1)[0]))
        stopwatch.start(file_name)

        # Extract Audio
        if not audio_flag:
            logger.info(f"Processing video file: {input_media_path}")
            audio_path: str = extract_audio(video_path=input_media_path)
        else:
            logger.info(f"Processing audio file: {input_media_path}")
            audio_path = input_media_path

        # Get model size
        model_size = get_model(model_size=args.model_size, language=args.language)

        # Transcribe audio
        language, segments = get_transcription(
            audio_path=audio_path,
            device=get_device(args.compute_device.lower()),
            model_size=model_size,
            language=args.language,
            num_threads=args.num_threads,
            print_progress=bool(print_progress),
        )

        # Generate unprocessed raw subtitles
        subtitles_raw: str = generate_subtitles(segments=segments)

        # Post-process subtitles
        subtitles: str = post_process(subtitles=subtitles_raw)

        # Write subtitles to file
        write_subtitles(
            subtitles=subtitles,
            file_name=file_name,
            input_media_path=input_media_path,
            language=language,
        )
        stopwatch.stop(file_name)

    # Print summary of processing times
    stopwatch.summary()


if __name__ == "__main__":
    import gc

    gc.collect()
    main()
    gc.collect()
