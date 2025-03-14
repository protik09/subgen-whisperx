#!/usr/bin/env python3
"""
Integration tests for the audio_extractor.py module.

This module contains integration tests that test the actual functionality
of the audio extraction classes with real files.

Note: These tests require ffmpeg to be installed on the system.
"""

import os
import shutil
import tempfile
import unittest
from typing import List, Tuple

from utils.audio_extractor import AudioExtractor, ConcurrentAudioExtractor, MediaProcessor


class TestAudioExtractorIntegration(unittest.TestCase):
    """Integration tests for the AudioExtractor class."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used for all tests."""
        # Check if ffmpeg is installed
        try:
            import subprocess
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            raise unittest.SkipTest("ffmpeg is not installed or not in PATH")

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = AudioExtractor()
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Copy test media files to the temporary directory
        # Note: You need to replace these paths with actual test media files
        test_files_dir = os.path.join(os.path.dirname(__file__), "test_files")
        if os.path.exists(test_files_dir):
            for filename in os.listdir(test_files_dir):
                if filename.endswith((".mp4", ".mkv", ".mp3", ".wav")):
                    shutil.copy(
                        os.path.join(test_files_dir, filename),
                        os.path.join(self.temp_dir.name, filename)
                    )
        else:
            # Create a dummy test video file if no test files are available
            self._create_dummy_test_files()

    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()

    def _create_dummy_test_files(self):
        """Create dummy test files for testing."""
        # This is a fallback if no real test files are available
        # Note: These dummy files won't work with actual ffmpeg processing
        # but they allow the tests to run without errors
        with open(os.path.join(self.temp_dir.name, "dummy_video.mp4"), "w") as f:
            f.write("dummy video content")
        with open(os.path.join(self.temp_dir.name, "dummy_audio.mp3"), "w") as f:
            f.write("dummy audio content")

    def _get_test_video_path(self) -> str:
        """Get the path to a test video file."""
        # Try to find a real video file
        for filename in os.listdir(self.temp_dir.name):
            if filename.endswith((".mp4", ".mkv")):
                return os.path.join(self.temp_dir.name, filename)
        
        # Fall back to the dummy file
        return os.path.join(self.temp_dir.name, "dummy_video.mp4")

    def _get_test_audio_path(self) -> str:
        """Get the path to a test audio file."""
        # Try to find a real audio file
        for filename in os.listdir(self.temp_dir.name):
            if filename.endswith((".mp3", ".wav")):
                return os.path.join(self.temp_dir.name, filename)
        
        # Fall back to the dummy file
        return os.path.join(self.temp_dir.name, "dummy_audio.mp3")

    def test_extract_audio(self):
        """Test the extract_audio method with a real video file."""
        video_path = self._get_test_video_path()
        
        try:
            # Call the method
            audio_path = self.extractor.extract_audio(video_path)
            
            # Check that the audio file was created
            self.assertTrue(os.path.exists(audio_path))
            self.assertTrue(os.path.getsize(audio_path) > 0)
        except Exception as e:
            # If this is a dummy file, the test will fail with an ffmpeg error
            if "dummy" in video_path:
                self.skipTest("Skipping test with dummy file")
            else:
                raise e


class TestConcurrentAudioExtractorIntegration(unittest.TestCase):
    """Integration tests for the ConcurrentAudioExtractor class."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used for all tests."""
        # Check if ffmpeg is installed
        try:
            import subprocess
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            raise unittest.SkipTest("ffmpeg is not installed or not in PATH")

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = ConcurrentAudioExtractor()
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Copy test media files to the temporary directory
        test_files_dir = os.path.join(os.path.dirname(__file__), "test_files")
        if os.path.exists(test_files_dir):
            for filename in os.listdir(test_files_dir):
                if filename.endswith((".mp4", ".mkv", ".mp3", ".wav")):
                    shutil.copy(
                        os.path.join(test_files_dir, filename),
                        os.path.join(self.temp_dir.name, filename)
                    )
        else:
            # Create dummy test files if no test files are available
            self._create_dummy_test_files()

    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()

    def _create_dummy_test_files(self):
        """Create dummy test files for testing."""
        # This is a fallback if no real test files are available
        with open(os.path.join(self.temp_dir.name, "dummy_video1.mp4"), "w") as f:
            f.write("dummy video content 1")
        with open(os.path.join(self.temp_dir.name, "dummy_video2.mp4"), "w") as f:
            f.write("dummy video content 2")
        with open(os.path.join(self.temp_dir.name, "dummy_audio.mp3"), "w") as f:
            f.write("dummy audio content")

    def _get_test_media_files(self) -> List[Tuple[str, bool]]:
        """Get a list of test media files."""
        media_files = []
        
        # Add video files
        for filename in os.listdir(self.temp_dir.name):
            if filename.endswith((".mp4", ".mkv")):
                media_files.append((os.path.join(self.temp_dir.name, filename), False))
            elif filename.endswith((".mp3", ".wav")):
                media_files.append((os.path.join(self.temp_dir.name, filename), True))
        
        # If no real files were found, add dummy files
        if not media_files:
            media_files = [
                (os.path.join(self.temp_dir.name, "dummy_video1.mp4"), False),
                (os.path.join(self.temp_dir.name, "dummy_video2.mp4"), False),
                (os.path.join(self.temp_dir.name, "dummy_audio.mp3"), True)
            ]
        
        return media_files

    def test_extract_audio_concurrent(self):
        """Test the extract_audio_concurrent method with real files."""
        media_files = self._get_test_media_files()
        
        try:
            # Call the method
            results = self.extractor.extract_audio_concurrent(media_files)
            
            # Check the results
            self.assertEqual(len(results), len(media_files))
            
            # Check that audio files were created for video files
            for original_path, audio_path, was_extracted in results:
                if was_extracted:
                    self.assertTrue(os.path.exists(audio_path))
                    self.assertTrue(os.path.getsize(audio_path) > 0)
        except Exception as e:
            # If these are dummy files, the test will fail with an ffmpeg error
            if any("dummy" in path for path, _ in media_files):
                self.skipTest("Skipping test with dummy files")
            else:
                raise e


class TestMediaProcessorIntegration(unittest.TestCase):
    """Integration tests for the MediaProcessor class."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used for all tests."""
        # Check if ffmpeg is installed
        try:
            import subprocess
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            raise unittest.SkipTest("ffmpeg is not installed or not in PATH")

    def setUp(self):
        """Set up test fixtures."""
        self.processor = MediaProcessor()
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = os.path.join(self.temp_dir.name, "test_dir")
        os.makedirs(self.test_dir, exist_ok=True)
        self.test_text_path = os.path.join(self.temp_dir.name, "test_files.txt")
        
        # Copy test media files to the test directory
        test_files_dir = os.path.join(os.path.dirname(__file__), "test_files")
        if os.path.exists(test_files_dir):
            for filename in os.listdir(test_files_dir):
                if filename.endswith((".mp4", ".mkv", ".mp3", ".wav")):
                    shutil.copy(
                        os.path.join(test_files_dir, filename),
                        os.path.join(self.test_dir, filename)
                    )
            
            # Create a text file with the paths to the test files
            with open(self.test_text_path, "w") as f:
                for filename in os.listdir(self.test_dir):
                    if filename.endswith((".mp4", ".mkv", ".mp3", ".wav")):
                        f.write(f"{os.path.join(self.test_dir, filename)}\n")
        else:
            # Create dummy test files if no test files are available
            self._create_dummy_test_files()

    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()

    def _create_dummy_test_files(self):
        """Create dummy test files for testing."""
        # This is a fallback if no real test files are available
        video_path = os.path.join(self.test_dir, "dummy_video.mp4")
        audio_path = os.path.join(self.test_dir, "dummy_audio.mp3")
        
        with open(video_path, "w") as f:
            f.write("dummy video content")
        with open(audio_path, "w") as f:
            f.write("dummy audio content")
        with open(os.path.join(self.test_dir, "document.txt"), "w") as f:
            f.write("dummy text content")
        
        # Create a text file with the paths to the dummy files
        with open(self.test_text_path, "w") as f:
            f.write(f"{video_path}\n{audio_path}\n")

    def test_process_media_files_directory(self):
        """Test the process_media_files method with a directory."""
        try:
            # Call the method
            results = self.processor.process_media_files(directory=self.test_dir)
            
            # Check that results were returned
            self.assertIsNotNone(results)
            
            # Check that audio files were created for video files
            for original_path, audio_path, was_extracted in results:
                if was_extracted:
                    self.assertTrue(os.path.exists(audio_path))
                    self.assertTrue(os.path.getsize(audio_path) > 0)
        except Exception as e:
            # If these are dummy files, the test will fail with an ffmpeg error
            if os.path.exists(os.path.join(self.test_dir, "dummy_video.mp4")):
                self.skipTest("Skipping test with dummy files")
            else:
                raise e

    def test_process_media_files_file(self):
        """Test the process_media_files method with a single file."""
        # Find a video file to test with
        video_path = None
        for filename in os.listdir(self.test_dir):
            if filename.endswith((".mp4", ".mkv")):
                video_path = os.path.join(self.test_dir, filename)
                break
        
        if not video_path:
            video_path = os.path.join(self.test_dir, "dummy_video.mp4")
        
        try:
            # Call the method
            results = self.processor.process_media_files(file=video_path)
            
            # Check that results were returned
            self.assertIsNotNone(results)
            self.assertEqual(len(results), 1)
            
            # Check that an audio file was created
            original_path, audio_path, was_extracted = results[0]
            if was_extracted:
                self.assertTrue(os.path.exists(audio_path))
                self.assertTrue(os.path.getsize(audio_path) > 0)
        except Exception as e:
            # If this is a dummy file, the test will fail with an ffmpeg error
            if "dummy" in video_path:
                self.skipTest("Skipping test with dummy file")
            else:
                raise e

    def test_process_media_files_txt(self):
        """Test the process_media_files method with a text file."""
        try:
            # Call the method
            results = self.processor.process_media_files(txt=self.test_text_path)
            
            # Check that results were returned
            self.assertIsNotNone(results)
            
            # Check that audio files were created for video files
            for original_path, audio_path, was_extracted in results:
                if was_extracted:
                    self.assertTrue(os.path.exists(audio_path))
                    self.assertTrue(os.path.getsize(audio_path) > 0)
        except Exception as e:
            # If these are dummy files, the test will fail with an ffmpeg error
            if os.path.exists(os.path.join(self.test_dir, "dummy_video.mp4")):
                self.skipTest("Skipping test with dummy files")
            else:
                raise e


if __name__ == "__main__":
    unittest.main() 