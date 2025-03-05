import argparse
import concurrent.futures
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

try:
    import coloredlogs
    import ffmpeg
    import srt  # TODO: Remove dependency in future update
except ImportError as e:
    import subprocess

    # Yes I'm aware this could be a potential security issue
    # Install missing dependencies
    print(f"Installing missing dependencies: {e}")
    if os.windows:
        commandline_options = [
            "powershell.exe",
            "-ExecutionPolicy",
            "Unrestricted",
            "uv_init.ps1",
        ]
    else:
        commandline_options = ["bash", "uv_init.sh"]
    process_result = subprocess.run(
        commandline_options,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
from utils.exceptions import FolderNotFoundError, MediaNotFoundError
import utils.timer as timer
from utils.constants import MEDIA_EXTENSIONS, MODELS_AVAILABLE, WHISPER_LANGUAGE

# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(
    log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_subgen.log"
)
LOGGING_LEVEL = logging.DEBUG
logging.basicConfig(
    filename=log_filename,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
    level=LOGGING_LEVEL,
)
coloredlogs.install(level=logging.getLevelName(LOGGING_LEVEL))

# Init global timer
stopwatch: timer.Timer = timer.Timer(logging.getLevelName(LOGGING_LEVEL))


# TODO: Clean this function up to be more specific rather than generic
# Improved DLL loading
def setup_dll_paths():
    """Set up paths for DLL loading"""
    logger = logging.getLogger("setup_dll_paths")

    # Add multiple potential DLL locations
    dll_paths = [
        # Current directory libs folder
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs"),
        # Executable directory libs folder (for PyInstaller)
        os.path.join(os.path.dirname(sys.executable), "libs"),
        # Current working directory libs folder
        os.path.join(os.getcwd(), "libs"),
    ]

    # Log the paths we're adding
    for path in dll_paths:
        if os.path.exists(path):
            logger.debug(f"Adding DLL path: {path}")
            if sys.platform == "win32":
                os.add_dll_directory(path)
            os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
        else:
            logger.debug(f"DLL path does not exist: {path}")

    # For CUDA specifically
    if sys.platform == "win32":
        # Try to find CUDA path from environment or common locations
        cuda_paths = [
            os.environ.get("CUDA_PATH"),
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.3",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.2",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8",
        ]

        for cuda_path in cuda_paths:
            if cuda_path and os.path.exists(cuda_path):
                bin_path = os.path.join(cuda_path, "bin")
                if os.path.exists(bin_path):
                    logger.debug(f"Adding CUDA bin path: {bin_path}")
                    os.add_dll_directory(bin_path)
                    os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
                break


# Call the setup function
setup_dll_paths()


def get_device(device_selection: str | None = None) -> str:
    """Determine the best available device with graceful fallback"""
    from torch import cuda

    logger = logging.getLogger("get_device")

    if device_selection is None or "cuda" in device_selection.lower():
        try:
            if cuda.is_available():
                logger.info("CUDA available.")
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
        return _valid_media_flag, _valid_audio_flag
    except Exception as e:
        logger.error(f"An error occurred while probing the file: {e}")
        return False, False


def get_media_files(
    directory: str | None = None, file: str | None = None, txt: str | None = None
) -> List[Tuple[str, bool]] | None:
    """Get list of valid media files from directory and/or single file."""
    logger = logging.getLogger("get_media_files")
    media_extensions = tuple(MEDIA_EXTENSIONS)

    # Use set to prevent duplicates
    potential_media_files = set()
    media_files: List[Tuple[str, bool]] = []

    # Collect all potential media files
    if file and os.path.isfile(file):
        potential_media_files.add(file)

    if directory and os.path.isdir(directory):
        for root, _, files in os.walk(directory):
            for f in files:
                if f.endswith(media_extensions):
                    potential_media_files.add(os.path.join(root, f))

    # Filter for file paths with valid media extensions``
    if txt and os.path.isfile(txt):
        try:
            with open(txt, "r") as f:
                for line in f:
                    line = line.strip()
                    if os.path.isfile(line):
                        potential_media_files.add(line)
                    elif os.path.isdir(line):
                        for root, _, files in os.walk(line):
                            for f in files:
                                if f.endswith(media_extensions):
                                    potential_media_files.add(os.path.join(root, f))
        except Exception as e:
            logger.error(f"An error occurred while reading the txt file: {e}")
            raise

    if not potential_media_files:
        logger.error("No potential media files found")
        return None

    # Validate files concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(is_media_file, file_path): file_path
            for file_path in potential_media_files
        }

        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                is_valid, is_audio = future.result()
                if is_valid:
                    media_files.append((file_path, is_audio))
                else:
                    logger.warning(f"Skipping invalid media file: '{file_path}'")
            except Exception as e:
                logger.error(f"Validation failed for {file_path}: {str(e)}")

    if not media_files:
        logger.error("No valid media files found")
        return None

    logger.debug(f"Valid media files discovered: {media_files}")
    return media_files


def extract_audio(video_path: str = "") -> str:
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
        logger.error(f"An error occurred while extracting audio: {str(e)}")
    stopwatch.stop("Audio Extraction")
    return extracted_audio_path


def extract_audio_concurrent(
    media_files: List[Tuple[str, bool]],
) -> List[Tuple[str, str, bool]]:
    """Extract audio from multiple files concurrently using multiprocessing.

    Args:
        media_files (List[Tuple[str, bool]]): List of (file_path, is_audio) tuples

    Returns:
        List[Tuple[str, str, bool]]: List of (original_path, audio_path, was_extracted) tuples
    """
    logger = logging.getLogger("extract_audio_concurrent")
    results = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_file = {}
        for file_path, is_audio in media_files:
            if not is_audio:
                # Only submit non-audio files for extraction
                future = executor.submit(extract_audio, file_path)
                future_to_file[future] = (file_path, is_audio)
            else:
                # Add audio files directly to results
                results.append((file_path, file_path, False))

        # Process completed extractions
        for future in concurrent.futures.as_completed(future_to_file):
            original_path, is_audio = future_to_file[future]
            try:
                audio_path = future.result()
                results.append((original_path, audio_path, True))
                logger.info(f"Successfully extracted audio from {original_path}")
            except Exception as e:
                logger.error(f"Failed to extract audio from {original_path}: {e}")
                results.append((original_path, original_path, False))

    return results


def get_raw_transcription(
    audio_paths: list[str],
    device: str,
    model_size: str,
    print_progress: bool = False,
    language: str | None = None,
    num_threads: int | None = None,
) -> list:
    stopwatch.start("Transcribe")
    import gc
    import torch
    import whisperx

    logger = logging.getLogger("transcribe")

    # TODO: Clean up the following spagetti code with something cleaner
    # Set number of threads for transcription
    threads_available: int | None = os.cpu_count()
    if threads_available is None and num_threads is None:
        threads_available = 1
    elif num_threads is None or num_threads < 1 or num_threads > threads_available:
        num_threads = threads_available
    else:
        pass
    pass

    # Load model
    model = whisperx.load_model(
        whisper_arch=model_size,
        device=device,
        compute_type="int8",
        language=language,
        threads=num_threads,
    )
    # Transcribe all audio
    initial_transcripts: list = []
    for audio_path in audio_paths:
        try:
            # Initial transcription
            initial_transcripts.append(
                model.transcribe(
                    audio=audio_path,
                    batch_size=16,
                    print_progress=print_progress,
                )
            )
            logger.info(f"Transcribed: {os.path.basename(audio_path)}")
        except Exception as e:
            logger.error(f"Transcription failure: {os.path.basename(audio_path)}.")
            logger.debug(f"{e}")

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()
    stopwatch.stop("Transcribe")
    return initial_transcripts


def get_aligned_transcripts(
    language: str,
    device: str,
    initial_transcript: list,
    audio_paths: list[str],
    print_progress: bool = False,
) -> Tuple[str, dict]:
    stopwatch.start("Align Transcripts")
    import torch
    import whisperx

    logger = logging.getLogger("align_transcripts")

    # Store language before alignment
    if language is None:
        language = initial_transcript["language"]

    # Load Alignment Model
    model_a, metadata = whisperx.load_align_model(
        language_code=language,
        device=device,
    )
    aligned_transcripts: list = []
    # Align all the text and audio data
    for audio_path in audio_paths:
        try:
            aligned_transcripts.append(
                whisperx.align(
                    transcript=initial_transcript["segments"],
                    model=model_a,
                    align_model_metadata=metadata,
                    audio=audio_path,
                    device=device,
                    print_progress=print_progress,
                )
            )
        except Exception as e:
            logger.error(f"Alignment failure: {os.path.basename(audio_path)}.")
            logger.debug(f"{e}")

    # Delete CUDA cache
    del model_a
    torch.cuda.empty_cache()
    gc.collect()
    segments = []
    languages = []
    # Get aligned segments
    for transcribed_segments in aligned_transcripts:
        segments.append(transcribed_segments["segments"])
        languages.append(transcribed_segments["language"])

    stopwatch.stop("Align Transcripts")
    return languages, segments


def get_transcription(
    audio_path: str,
    device: str,
    model_size: str,
    print_progress: bool = False,
    language: str | None = None,
    num_threads: int | None = None,
) -> Tuple[str | None, List[Dict[str, float | str]] | None]:
    """
    Transcribes audio file using WhisperX model and aligns timestamps.
    It handles model loading, transcription, and alignment in one workflow.
    Args:
        audio_path (str): Path to the audio file to transcribe
        device (str): Device to run inference on ('cuda' or 'cpu')
        model_size (str): Size of the Whisper model to use (e.g. 'tiny', 'base', 'small', 'medium', 'large')
        print_progress (bool, optional):Whether to print progress during transcripton. Defaults to False.
        language (str, optional): Language code for transcription. If None, auto-detects language. Defaults to None.
        num_threads (int, optional): Number of CPU threads to use. If None, uses all available threads. Defaults to None.
    Returns:
        Tuple[str | None, List[Dict[str, float | str]] | None]: A tuple containing:
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
    import gc

    import torch

    import whisperx

    logger = logging.getLogger("Transcription")
    stopwatch.start("Transcription")

    # TODO: Clean up the following spagetti code with something cleaner
    # Set number of threads for transcription
    threads_available: int | None = os.cpu_count()
    if threads_available is None and num_threads is None:
        threads_available = 1
    elif num_threads is None or num_threads < 1 or num_threads > threads_available:
        num_threads = threads_available
    else:
        pass

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
    # Try nd free GPU memory for next model load
    del model
    torch.cuda.empty_cache()
    gc.collect()
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

    # Delete CUDA cache
    del model_a
    torch.cuda.empty_cache()
    gc.collect()

    # Get aligned segments
    segments: List[Dict[str, float | str]] = aligned_result["segments"]

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

        # SRT format: [segment number]\n[start] --> [end]\n[text]\n
        _srt_content.append(f"\n{os.linesep}{i}")
        _srt_content.append(f"{segment_start} --> {segment_end}")
        _srt_content.append(f"{text}")

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

    return _subtitles_clean.lstrip()


def write_subtitles(
    subtitles: str, file_name: str, input_media_path: str, language: str | None
) -> None:
    """Write the generated subtitles to a file.
    Args:
        subtitles (str): The generated subtitles as a single string
        output_path (str): The path to write the subtitles to
        language (str): The language for subtitles
    """
    logger = logging.getLogger("write_subtitles")
    # The following should generate something like "input.ai.srt" from "input.mp4"
    _subtitle_file_name = f"{file_name}.{language}-AI.srt"

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
    args = parser.parse_args()

    # Set logging level
    logging_level = getattr(logging, args.log_level.upper(), logging.DEBUG)
    logging.getLogger().setLevel(logging_level)
    coloredlogs.install(level=logging_level)
    # Set print_prgress flag depending on logging level
    print_progress = logging_level < logging.INFO

    # If no args are passed to argparser, print help and exit
    if len(sys.argv) == 1:
        parser.print_help(sys.stdout)
        sys.exit()

    # Check that args.directory is a valid directory only if specified in the arguments
    if args.directory and not os.path.isdir(args.directory):
        logger.error(f"Error: Directory '{args.directory}' does not exist.")
        raise FolderNotFoundError
    # Check that args.file is a valid file only if specified in the arguments
    if args.file and not os.path.isfile(args.file):
        logger.error(f"Error: File '{args.file}' does not exist.")
        raise FileNotFoundError

    # Check that the language flag passed is compatible with Whisper
    if args.language and args.language not in WHISPER_LANGUAGE:
        logger.error(
            f"The language code {args.language} is not a valid ISO 639-1 code supported by Whisper"
        )
        raise KeyError

    # Get only the media files from the paths given
    media_files = get_media_files(args.directory, args.file, args.txt)
    if not media_files:
        logger.error("No media files found in the path provided")
        raise MediaNotFoundError

    try:
        # Extract audio from all files concurrently
        logger.info("Starting concurrent audio extraction...")
        extracted_files = extract_audio_concurrent(media_files)

        # Get model size
        model_size = get_model(model_size=args.model_size, language=args.language)

        for original_path, audio_path, was_extracted in extracted_files:
            file_name = str(os.path.basename(original_path.rsplit(".", 1)[0]))
            stopwatch.start(file_name)

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
                input_media_path=original_path,
                language=language,
            )

            # Cleanup extracted audio if needed
            if was_extracted:
                try:
                    os.remove(audio_path)
                    logger.debug(f"Cleaned up temporary audio file: {audio_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary audio file: {e}")

            stopwatch.stop(file_name)

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")
        stopwatch.stop(file_name)
        raise

    # Print summary of processing times
    stopwatch.summary()


if __name__ == "__main__":
    import gc

    gc.collect()
    main()
    gc.collect()
