import math
import time
from typing import Dict


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
