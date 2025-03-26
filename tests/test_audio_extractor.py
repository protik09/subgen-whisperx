#!/usr/bin/env python3
"""
Unit tests for the audio_extractor.py module.

This module contains tests for the AudioExtractor, ConcurrentAudioExtractor,
and MediaProcessor classes.
"""

import os
import sys
import tempfile
import unittest
import logging
from unittest.mock import patch, MagicMock

import ffmpeg

from objects.audio_extractor import AudioExtractor, ConcurrentAudioExtractor, MediaProcessor

# Set up logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger('test_audio_extractor')


class TestAudioExtractor(unittest.TestCase):
    """Tests for the AudioExtractor class."""

    def setUp(self):
        """Set up test fixtures."""
        logger.debug("Setting up TestAudioExtractor")
        self.extractor = AudioExtractor()
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_video_path = os.path.join(self.temp_dir.name, "test_video.mp4")
        # Create an empty file for testing
        with open(self.test_video_path, "w") as f:
            f.write("dummy content")
        logger.debug(f"Created test file at {self.test_video_path}")

    def tearDown(self):
        """Tear down test fixtures."""
        logger.debug("Tearing down TestAudioExtractor")
        self.temp_dir.cleanup()
        logger.debug("Temporary directory cleaned up")

    @patch("ffmpeg.input")
    @patch("ffmpeg.output")
    @patch("ffmpeg.run")
    @patch("utils.timer.Timer.start")
    @patch("utils.timer.Timer.stop")
    def test_extract_audio(self, mock_timer_stop, mock_timer_start, mock_ffmpeg_run, 
                          mock_ffmpeg_output, mock_ffmpeg_input):
        """Test the extract_audio method."""
        logger.debug("Running test_extract_audio")
        # Set up mocks
        mock_ffmpeg_input.return_value = "input_stream"
        mock_ffmpeg_output.return_value = "output_stream"
        
        # Call the method
        logger.debug("Calling extract_audio")
        result = self.extractor.extract_audio(self.test_video_path)
        logger.debug(f"extract_audio returned: {result}")
        
        # Check that the mocks were called correctly
        mock_timer_start.assert_called_once()
        mock_ffmpeg_input.assert_called_once_with(self.test_video_path)
        mock_ffmpeg_output.assert_called_once()
        mock_ffmpeg_run.assert_called_once_with("output_stream", overwrite_output=True)
        mock_timer_stop.assert_called_once()
        
        # Check the result
        expected_output = "audio-test_video.mp3"
        self.assertEqual(result, expected_output)
        logger.debug("test_extract_audio completed successfully")

    @patch("ffmpeg.input")
    @patch("ffmpeg.output")
    @patch("ffmpeg.run")
    def test_extract_audio_exception(self, mock_ffmpeg_run, mock_ffmpeg_output, mock_ffmpeg_input):
        """Test the extract_audio method when an exception occurs."""
        logger.debug("Running test_extract_audio_exception")
        # Set up mocks
        mock_ffmpeg_input.return_value = "input_stream"
        mock_ffmpeg_output.return_value = "output_stream"
        mock_ffmpeg_run.side_effect = Exception("ffmpeg error")
        
        # Call the method and check that it raises an exception
        with self.assertRaises(Exception):
            logger.debug("Calling extract_audio (expecting exception)")
            self.extractor.extract_audio(self.test_video_path)
        logger.debug("test_extract_audio_exception completed successfully")


class TestConcurrentAudioExtractor(unittest.TestCase):
    """Tests for the ConcurrentAudioExtractor class."""

    def setUp(self):
        """Set up test fixtures."""
        logger.debug("Setting up TestConcurrentAudioExtractor")
        # Patch the AudioExtractor to avoid actual extraction
        patcher = patch.object(AudioExtractor, 'extract_audio')
        self.mock_extract_audio = patcher.start()
        self.addCleanup(patcher.stop)
        self.mock_extract_audio.return_value = "audio-test_video.mp3"
        
        self.extractor = ConcurrentAudioExtractor()
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_video_path = os.path.join(self.temp_dir.name, "test_video.mp4")
        self.test_audio_path = os.path.join(self.temp_dir.name, "test_audio.mp3")
        # Create empty files for testing
        with open(self.test_video_path, "w") as f:
            f.write("dummy video content")
        with open(self.test_audio_path, "w") as f:
            f.write("dummy audio content")
        logger.debug(f"Created test files at {self.test_video_path} and {self.test_audio_path}")

    def tearDown(self):
        """Tear down test fixtures."""
        logger.debug("Tearing down TestConcurrentAudioExtractor")
        self.temp_dir.cleanup()
        logger.debug("Temporary directory cleaned up")

    @patch("concurrent.futures.ProcessPoolExecutor")
    @patch("concurrent.futures.as_completed")
    def test_extract_audio_concurrent(self, mock_as_completed, mock_executor_class):
        """Test the extract_audio_concurrent method."""
        logger.debug("Running test_extract_audio_concurrent")
        # Set up mocks
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        
        # Set up the future mock
        mock_future = MagicMock()
        mock_future.result.return_value = "audio-test_video.mp3"
        
        # Configure the submit method to return our mock future
        mock_executor.submit.return_value = mock_future
        
        # Set up as_completed to return our mock future
        mock_as_completed.return_value = [mock_future]
        
        # Create a list of test media files
        media_files = [
            (self.test_video_path, False),  # Video file
            (self.test_audio_path, True)    # Audio file
        ]
        
        # Call the method
        logger.debug("Calling extract_audio_concurrent")
        results = self.extractor.extract_audio_concurrent(media_files)
        logger.debug(f"extract_audio_concurrent returned: {results}")
        
        # Check that the executor was used correctly
        mock_executor.submit.assert_called_once()
        
        # Check the results
        self.assertEqual(len(results), 2)
        # Audio file (not extracted) should be in results
        self.assertTrue(any(item[0] == self.test_audio_path and item[1] == self.test_audio_path and item[2] == False for item in results))
        # Video file (extracted) should be in results
        self.assertTrue(any(item[0] == self.test_video_path and item[1] == "audio-test_video.mp3" and item[2] == True for item in results))
        logger.debug("test_extract_audio_concurrent completed successfully")

    @patch("concurrent.futures.ProcessPoolExecutor")
    @patch("concurrent.futures.as_completed")
    def test_extract_audio_concurrent_exception(self, mock_as_completed, mock_executor_class):
        """Test the extract_audio_concurrent method when an exception occurs."""
        logger.debug("Running test_extract_audio_concurrent_exception")
        # Set up mocks
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        
        # Set up the future mock to raise an exception
        mock_future = MagicMock()
        mock_future.result.side_effect = Exception("extraction error")
        
        # Configure the submit method to return our mock future
        mock_executor.submit.return_value = mock_future
        
        # Set up as_completed to return our mock future
        mock_as_completed.return_value = [mock_future]
        
        # Create a list of test media files
        media_files = [(self.test_video_path, False)]  # Video file
        
        # Call the method
        logger.debug("Calling extract_audio_concurrent (expecting exception handling)")
        results = self.extractor.extract_audio_concurrent(media_files)
        logger.debug(f"extract_audio_concurrent returned: {results}")
        
        # Check the results
        expected_results = [(self.test_video_path, self.test_video_path, False)]  # Failed extraction
        self.assertEqual(results, expected_results)
        logger.debug("test_extract_audio_concurrent_exception completed successfully")


class TestMediaProcessor(unittest.TestCase):
    """Tests for the MediaProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        logger.debug("Setting up TestMediaProcessor")
        # Patch the ConcurrentAudioExtractor to avoid hanging
        patcher = patch.object(ConcurrentAudioExtractor, 'extract_audio_concurrent')
        self.mock_extract_audio_concurrent = patcher.start()
        self.addCleanup(patcher.stop)
        
        # Set up a default return value for the mock
        self.mock_extract_audio_concurrent.return_value = []
        
        self.processor = MediaProcessor()
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_video_path = os.path.join(self.temp_dir.name, "test_video.mp4")
        self.test_audio_path = os.path.join(self.temp_dir.name, "test_audio.mp3")
        self.test_text_path = os.path.join(self.temp_dir.name, "test_files.txt")
        self.test_dir = os.path.join(self.temp_dir.name, "test_dir")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create empty files for testing
        with open(self.test_video_path, "w") as f:
            f.write("dummy video content")
        with open(self.test_audio_path, "w") as f:
            f.write("dummy audio content")
        with open(self.test_text_path, "w") as f:
            f.write(f"{self.test_video_path}\n{self.test_audio_path}")
        
        # Create test files in the test directory
        with open(os.path.join(self.test_dir, "test_video.mp4"), "w") as f:
            f.write("dummy video content")
        with open(os.path.join(self.test_dir, "test_audio.mp3"), "w") as f:
            f.write("dummy audio content")
        with open(os.path.join(self.test_dir, "document.txt"), "w") as f:
            f.write("dummy text content")
        logger.debug("Test files created")

    def tearDown(self):
        """Tear down test fixtures."""
        logger.debug("Tearing down TestMediaProcessor")
        self.temp_dir.cleanup()
        logger.debug("Temporary directory cleaned up")

    def test_is_media_file(self):
        """Test the is_media_file method."""
        logger.debug("Running test_is_media_file")
        # Test with video file
        is_media, is_audio = self.processor.is_media_file(self.test_video_path)
        self.assertTrue(is_media)
        self.assertFalse(is_audio)
        
        # Test with audio file
        is_media, is_audio = self.processor.is_media_file(self.test_audio_path)
        self.assertTrue(is_media)
        self.assertTrue(is_audio)
        
        # Test with non-media file
        is_media, is_audio = self.processor.is_media_file(self.test_text_path)
        self.assertFalse(is_media)
        self.assertFalse(is_audio)
        
        # Test with non-existent file
        is_media, is_audio = self.processor.is_media_file("non_existent_file.mp4")
        self.assertFalse(is_media)
        self.assertFalse(is_audio)
        logger.debug("test_is_media_file completed successfully")

    def test_get_media_files_directory(self):
        """Test the get_media_files method with a directory."""
        logger.debug("Running test_get_media_files_directory")
        # Call the method
        media_files = self.processor.get_media_files(directory=self.test_dir)
        
        # Check the results
        self.assertEqual(len(media_files), 2)  # 2 media files in the directory
        
        # Check that the media files are correctly identified
        paths = [path for path, _ in media_files]
        self.assertIn(os.path.join(self.test_dir, "test_video.mp4"), paths)
        self.assertIn(os.path.join(self.test_dir, "test_audio.mp3"), paths)
        
        # Check that non-media files are not included
        self.assertNotIn(os.path.join(self.test_dir, "document.txt"), paths)
        logger.debug("test_get_media_files_directory completed successfully")

    def test_get_media_files_file(self):
        """Test the get_media_files method with a single file."""
        logger.debug("Running test_get_media_files_file")
        # Call the method with a video file
        media_files = self.processor.get_media_files(file=self.test_video_path)
        
        # Check the results
        self.assertEqual(len(media_files), 1)
        self.assertEqual(media_files[0], (self.test_video_path, False))
        
        # Call the method with an audio file
        media_files = self.processor.get_media_files(file=self.test_audio_path)
        
        # Check the results
        self.assertEqual(len(media_files), 1)
        self.assertEqual(media_files[0], (self.test_audio_path, True))
        
        # Call the method with a non-media file
        try:
            from utils.exceptions import MediaNotFoundError
            with self.assertRaises(MediaNotFoundError):
                self.processor.get_media_files(file=self.test_text_path)
        except ImportError:
            logger.warning("Could not import MediaNotFoundError, skipping part of the test")
        logger.debug("test_get_media_files_file completed successfully")

    def test_get_media_files_txt(self):
        """Test the get_media_files method with a text file."""
        logger.debug("Running test_get_media_files_txt")
        # Call the method
        media_files = self.processor.get_media_files(txt=self.test_text_path)
        
        # Check the results
        self.assertEqual(len(media_files), 2)
        self.assertIn((self.test_video_path, False), media_files)
        self.assertIn((self.test_audio_path, True), media_files)
        logger.debug("test_get_media_files_txt completed successfully")

    def test_get_media_files_nonexistent_directory(self):
        """Test the get_media_files method with a non-existent directory."""
        logger.debug("Running test_get_media_files_nonexistent_directory")
        try:
            from utils.exceptions import FolderNotFoundError
            with self.assertRaises(FolderNotFoundError):
                self.processor.get_media_files(directory="non_existent_directory")
        except ImportError:
            logger.warning("Could not import FolderNotFoundError, skipping test")
        logger.debug("test_get_media_files_nonexistent_directory completed successfully")

    def test_get_media_files_nonexistent_txt(self):
        """Test the get_media_files method with a non-existent text file."""
        logger.debug("Running test_get_media_files_nonexistent_txt")
        try:
            from utils.exceptions import MediaNotFoundError
            with self.assertRaises(MediaNotFoundError):
                self.processor.get_media_files(txt="non_existent_file.txt")
        except ImportError:
            logger.warning("Could not import MediaNotFoundError, skipping test")
        logger.debug("test_get_media_files_nonexistent_txt completed successfully")

    def test_get_media_files_empty_directory(self):
        """Test the get_media_files method with an empty directory."""
        logger.debug("Running test_get_media_files_empty_directory")
        # Create an empty directory
        empty_dir = os.path.join(self.temp_dir.name, "empty_dir")
        os.makedirs(empty_dir, exist_ok=True)
        
        # Call the method
        try:
            from utils.exceptions import MediaNotFoundError
            with self.assertRaises(MediaNotFoundError):
                self.processor.get_media_files(directory=empty_dir)
        except ImportError:
            logger.warning("Could not import MediaNotFoundError, skipping test")
        logger.debug("test_get_media_files_empty_directory completed successfully")

    def test_process_media_files(self):
        """Test the process_media_files method."""
        logger.debug("Running test_process_media_files")
        # Set up mock
        self.mock_extract_audio_concurrent.return_value = [
            (self.test_video_path, "audio-test_video.mp3", True),
            (self.test_audio_path, self.test_audio_path, False)
        ]
        
        # Call the method with a directory
        results = self.processor.process_media_files(directory=self.test_dir)
        
        # Check that the mock was called correctly
        self.mock_extract_audio_concurrent.assert_called_once()
        
        # Check the results
        self.assertEqual(results, self.mock_extract_audio_concurrent.return_value)
        logger.debug("test_process_media_files completed successfully")


if __name__ == "__main__":
    # Use a test runner that doesn't capture stdout/stderr
    # This helps with debugging hanging tests
    unittest.main(buffer=False) 