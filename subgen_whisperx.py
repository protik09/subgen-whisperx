import sys
import os
import time
import math
import ffmpeg
import whisperx
import torch

from typing import List, Dict, Any
from torch import cuda

DEFAULT_INPUT_VIDEO = "input.mp4"
INPUT_VIDEO_NAME = DEFAULT_INPUT_VIDEO.replace(".mp4", "")

# WhisperX supports these models
MODELS_AVAILABLE: List[str] = [
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
]
MODE_SIZE = "base.en"
device = "cuda" if cuda.is_available() else "cpu"


def format_time(seconds: float) -> str:
    """
    Converts a time duration given in seconds to a formatted string in the format HH:MM:SS,mmm.

    Parameters:
    seconds (float): The total duration in seconds, including fractional seconds.

    Returns:
    str: A formatted string representing the time duration in the format HH:MM:SS,mmm.
    """
    hours: int = math.floor(seconds / 3600)
    seconds %= 3600
    minutes: int = math.floor(seconds / 60)
    seconds %= 60
    milliseconds: int = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    formatted_time: str = f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"  # SRT timing format
    return formatted_time


class Timer:
    """
    A utility class for measuring and tracking execution times of different operations.
    This class provides functionality to measure the duration of named operations
    by tracking their start and end times. It maintains a dictionary of timings
    that can be used to analyze performance.
    Attributes:
        timings (Dict[str, Dict[str, float]]): A nested dictionary storing timing information.
            The outer dictionary uses operation names as keys, while the inner dictionary
            contains 'start', 'end', and 'duration' timestamps for each operation.
    Methods:
        start(name: str): Starts the timer for a named operation
        stop(name: str): Stops the timer for a named operation and calculates duration
        summary(): Prints a summary of all timed operations and their durations
    Example:
        >>> timer = Timer()
        >>> timer.start("operation1")
        >>> # ... some code to time ...
        >>> timer.stop("operation1")
        >>> timer.summary()
        === Processing Times ===
        operation1: 1.23s
        Total time: 1.23s
    Note:
        - The timer uses time.time() for measurements
        - All times are in seconds
        - If stop() is called for an operation that wasn't started, a warning is printed
        - The summary() method formats times using a separate format_time() function
    """

    def __init__(self) -> None:
        self.timings: Dict[str, Dict[str, float]] = {}

    def start(self, name: str) -> None:
        self.timings[name] = {"start": time.time()}

    def stop(self, name: str) -> None:
        if name in self.timings:
            self.timings[name]["end"] = time.time()
            self.timings[name]["duration"] = (
                self.timings[name]["end"] - self.timings[name]["start"]
            )
        else:
            print(f"Warning: Timer for '{name}' was not started.")

    def summary(self) -> None:
        total: float = 0
        print("\n=== Processing Times ===")
        for name, timing in self.timings.items():
            duration: float = timing["duration"]
            total += duration
            print(f"{name}: {format_time(duration).replace(',', '.')} s")
        print(f"Total time: {format_time(total).replace(',', '.')} s")


# Init global timer
timer: Timer = Timer()


def get_device():
    """Determine the best available device with graceful fallback"""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        else:
            print("CUDA not available, falling back to CPU")
            return "cpu"
    except Exception as e:
        print(f"Warning: Error checking CUDA availability ({str(e)})")
        print("Falling back to CPU")
        return "cpu"


def extract_audio(video_path: str = DEFAULT_INPUT_VIDEO) -> str:
    timer.start("Audio Extraction")
    extracted_audio = f"audio-{INPUT_VIDEO_NAME}.mp3"

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
    """
    timer.start("Audio Extraction")
    extracted_audio: str = f"audio-{INPUT_VIDEO_NAME}.mp3"

    # Add optimization flags to ffmpeg
    stream = ffmpeg.input(video_path)
    stream = ffmpeg.output(
        stream,
        extracted_audio,
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
    timer.stop("Audio Extraction")
    return extracted_audio


def transcribe(audio):
    timer.start("Transcription")

    # Load model
    model = whisperx.load_model(MODE_SIZE, device)

    # Initial transcription
    initial_result = model.transcribe(audio, batch_size=16)

    # Store language before alignment
    language = initial_result["language"]

    # Align timestamps
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    aligned_result = whisperx.align(
        initial_result["segments"], model_a, metadata, audio, device
    )

    # Get aligned segments
    segments = aligned_result["segments"]

    print(f"Language: {language}")
    for segment in segments:
        print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")

    timer.stop("Transcription")
    return language, segments


def generate_subtitle_file(language, segments):
    srt_content = []
    for i, segment in enumerate(segments, start=1):
        segment_start = format_time(segment["start"])
        segment_end = format_time(segment["end"])
        text = segment["text"].strip()

        srt_content.append(f"{i}")
        srt_content.append(f"{segment_start} --> {segment_end}")
        srt_content.append(f"{text}\n")

    output_file = f"{INPUT_VIDEO_NAME}.whisperx.srt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_content))

    return output_file


# Function to accept file path from commandline argument
def get_input_video():
    if len(sys.argv) != 2:
        # Use the debugging path if no input video is provided

        print("No input video provided. Using debugging video path.")
        return DEFAULT_INPUT_VIDEO
    else:
        return str(sys.argv[1])


def main():
    input_video_path: str = get_input_video()
    # Check if the input video exists
    if not os.path.exists(input_video_path):
        print(f"Error: Input video '{input_video_path}' not found.")
        raise FileNotFoundError
    if not input_video_path:
        print("Error: No valid input video provided.")
        return

    extracted_audio = extract_audio()
    language, segments = transcribe(audio=extracted_audio)
    subtitle_file = generate_subtitle_file(language=language, segments=segments)
    print(f"Subtitle file generated: {subtitle_file}")
    timer.summary()


if __name__ == "__main__":
    import gc

    gc.collect()
    main()
    gc.collect()
