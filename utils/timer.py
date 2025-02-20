import os
import math
import time
import logging
import coloredlogs
from typing import Dict


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
        format_time(seconds: float): Converts a time duration in seconds to SRT format
        summary(): Prints a summary of all timed operations and their durations
    """

    def __init__(self, logging_level="INFO") -> None:
        self.timings: Dict[str, Dict[str, float]] = {}
        # Name logger after the class object
        self.logger = logging.getLogger(self.__class__.__name__)
        coloredlogs.install(level=logging_level, logger=self.logger)

    def start(self, name: str) -> None:
        self.timings[name] = {"start": time.time()}
        self.logger.debug(f"Timer started for '{name}'")

    def stop(self, name: str) -> None:
        if name in self.timings:
            self.timings[name]["end"] = time.time()
            self.timings[name]["duration"] = (
                self.timings[name]["end"] - self.timings[name]["start"]
            )
            self.logger.debug(f"Timer stopped for '{name}'")
        else:
            self.logger.warning(f"Warning: Timer for '{name}' was not started.")

    @staticmethod
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

    def summary(self) -> None:
        total: float = 0
        self.logger.info("=== Processing Times ===")
        for name, timing in self.timings.items():
            if "duration" in timing:
                # Timer was properly stopped
                duration: float = timing["duration"]
            elif "start" in timing:
                # Timer was started but never stopped
                duration: float = time.time() - timing["start"]
                self.logger.warning(f"Timer '{name}' was never stopped")
            else:
                # Invalid timer state
                self.logger.error(f"Invalid timer state for '{name}'")
                continue
                
            total += duration
            self.logger.info(f"{name}: {self.format_time(duration).replace(',', '.')}s")
        self.logger.info(f"Total time: {self.format_time(total).replace(',', '.')}s")
