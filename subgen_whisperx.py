import sys
import os
import ffmpeg
import whisperx
import utils.timer as timer
import logging
import coloredlogs
from datetime import datetime
from typing import Dict
from torch import cuda
from utils.constants import DEFAULT_INPUT_VIDEO, MODEL_SIZE
import argparse

# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(
    log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_subgen.log"
)
LOGGING_LEVEL = ["DEBUG", logging.DEBUG]
logging.basicConfig(filename=log_filename, filemode="a", level=LOGGING_LEVEL[1])
coloredlogs.install(level=LOGGING_LEVEL[0])

# Init global timer
stopwatch: timer.Timer = timer.Timer(LOGGING_LEVEL[0])


# Function to accept file path from commandline argument
def get_input_video(video_path: str) -> str:
    logger = logging.getLogger("get_input_video")
    if len(sys.argv) != 2:
        logger.info(
            f"No input video provided. Using debugging video path: {DEFAULT_INPUT_VIDEO}"
        )
        return DEFAULT_INPUT_VIDEO
    else:
        # Check if the input video exists and is a video file
        if os.path.exists(video_path):
            if ffmpeg.probe(video_path)["streams"][0]["codec_type"] == "audio":
                raise ValueError
            elif ffmpeg.probe(video_path)["streams"][0]["codec_type"] == "video":
                return video_path
            else:
                logger.error(
                    f"Error: Input file '{video_path}' is neither a video nor an audio file."
                )
                raise TypeError
        else:
            logger.error(f"Error: Input video '{video_path}' does not exist.")
            raise FileNotFoundError


def get_device():
    logger = logging.getLogger("get_device")
    """Determine the best available device with graceful fallback"""
    # Check if an nVidia card is available
    # If nvida-smi is not available, it will fall back to CPU
    if os.system("nvidia-smi") == 0:
        logger.info("nVidia GPU detected.")
        if cuda.is_available():
            logger.info("CUDA available.")
            return "cuda"
        else:
            logger.warning("CUDA is not accessible on your nVidia GPU.")
            logger.warning(
                "Please refer to the CUDNN and CUDA installation guide at https://developer.nvidia.com/cudnn and https://developer.nvidia.com/cuda-downloads."
            )
            logger.warning("Using CPU instead.")
            return "cpu"
    else:
        logger.info("nVidia GPU not available.")

    try:
        if cuda.is_available():
            return "cuda"
        else:
            logger.warning("CUDA not available, falling back to CPU")
            return "cpu"
    except Exception as e:
        logger.error(f"Warning: Error checking CUDA availability ({str(e)})")
        logger.warning("Falling back to CPU.")


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
    stopwatch.stop("Audio Extraction")
    return extracted_audio_path


def transcribe(audio_path: str, device: str) -> Dict:
    logger = logging.getLogger("transcribe")
    stopwatch.start("Transcription")

    # Load model
    model = whisperx.load_model(MODEL_SIZE, device, compute_type="int8")

    # Initial transcription
    initial_result = model.transcribe(audio_path, batch_size=16)

    # Store language before alignment
    language = initial_result["language"]

    # Align timestamps
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    aligned_result = whisperx.align(
        initial_result["segments"], model_a, metadata, audio_path, device
    )

    # Get aligned segments
    segments = aligned_result["segments"]

    logger.info(f"Language: {language}")
    for segment in segments:
        logger.debug(
            f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}"
        )

    stopwatch.stop("Transcription")
    return language, segments


def generate_subtitles(segments: Dict) -> str:
    logger = logging.getLogger("generate_subtitles")
    srt_content = []
    for i, segment in enumerate(segments, start=1):
        segment_start = timer.Timer.format_time(segment["start"])
        segment_end = timer.Timer.format_time(segment["end"])
        text = segment["text"].strip()

        # SRT format: [segment number] [start] --> [end] [text]
        srt_content.append(f"{os.linesep}{i}")
        srt_content.append(f"{segment_start} --> {segment_end}")
        srt_content.append(f"{text}{os.linesep}")

    return f"{os.linesep}".join(srt_content)


def post_process(subtitles: str) -> str:
    logger = logging.getLogger("post_process")
    """Post-process the generated subtitles.
    This function performs additional processing on the generated subtitles to improve readability
    and ensure compliance with common subtitle standards. The post-processing steps include:
    - Removing duplicate lines
    - Removing empty lines
    - Removing leading/trailing whitespace
    - Normalizing line endings
    Args:
        subtitles (list): The generated subtitles as a list of strings
    Returns:
        str: The post-processed subtitles as a single string
    """

    # Clip lines that go over 150 characters taking into account word boundaries
    subtitles_clean: str = ""
    for line in subtitles:
        if len(line) > 150:
            line = line[:150].rsplit(" ", 1)[0]
        else:
            pass

        subtitles_clean += line

    return subtitles_clean


def main():
    logger = logging.getLogger("main")
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Subtitle Generator")
    parser.add_argument(
        "input_video",
        nargs="?",
        default=DEFAULT_INPUT_VIDEO,
        help="Path to the input video file",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    args = parser.parse_args()

    # Set logging level
    logging_level = getattr(logging, args.log_level.upper(), logging.DEBUG)
    logging.getLogger().setLevel(logging_level)
    coloredlogs.install(level=logging_level)

    audio_flag: bool = False
    input_media_path: str = args.input_video
    # Get video
    try:
        input_media_path: str = get_input_video(input_media_path)
        logger.info(f"Input video: {input_media_path}")
    except ValueError:
        input_media_path = str(sys.argv[1])
        if ffmpeg.probe(input_media_path)["format"]["format_name"] == "mp3,wav,flac":
            logger.info("The input file is an audio file")
            audio_flag = True
        else:
            raise ValueError("Unsupported audio format.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return

    # Extract Audio from Video if not an audio file
    if not audio_flag:
        audio_path: str = extract_audio(video_path=input_media_path)
    else:
        audio_path: str = input_media_path

    language, segments = transcribe(audio_path=audio_path, device=get_device())

    subtitles_raw: str = generate_subtitles(segments=segments)

    subtitles: str = post_process(subtitles=subtitles_raw)

    # The following should generate something like "input.ai.srt" from "input.mp4"
    subtitle_file_name = f"{os.path.basename(input_media_path.rsplit('.', 1)[0])}.ai-{language}.srt"

    # Write subtitles to file
    subtitle_path = os.path.join(os.path.dirname(input_media_path), subtitle_file_name)

    try:
        with open(subtitle_path, "w", encoding="utf-8") as f:
            f.write(subtitles)
            logger.info(f"Subtitle file generated: {subtitle_file_name}")
    except Exception as e:
        logger.error(f"An error occurred while writing the subtitle file: {e}")

    # Print summary of processing times
    stopwatch.summary()


if __name__ == "__main__":
    import gc

    gc.collect()
    main()
    gc.collect()
