import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from objects.media import Media
from objects.options import Options
from utils.exceptions import MediaNotFoundError


class TestMedia(unittest.TestCase):
    """Test cases for the Media class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock Options object
        self.mock_options = MagicMock(spec=Options)
        self.mock_options.file = None
        self.mock_options.directory = None
        self.mock_options.txt = None
        
        # Create the Media object with mock options
        self.media = Media(self.mock_options)

    def test_initialization(self):
        """Test the initialization of the Media class."""
        # Check that the Media object is initialized correctly
        self.assertEqual(self.media._options, self.mock_options)
        self.assertEqual(self.media._media_files, [])
        self.assertEqual(self.media._extracted_audio_paths, [])

    @patch('os.path.isfile')
    def test_collect_potential_files_single_file(self, mock_isfile):
        """Test collecting potential files with a single file."""
        # Setup
        mock_isfile.return_value = True
        self.mock_options.file = '/assets/input.mp4'
        self.mock_options.directory = None
        self.mock_options.txt = None

        # Execute
        result = self.media._collect_potential_files()

        # Assert
        self.assertEqual(len(result), 1)
        # Use Path objects for comparison to handle platform-specific path separators
        expected_path = Path('/assets/input.mp4')
        self.assertEqual(next(iter(result)), expected_path)

    @patch('os.path.isdir')
    @patch('os.walk')
    def test_collect_potential_files_directory(self, mock_walk, mock_isdir):
        """Test collecting potential files from a directory."""
        # Setup
        mock_isdir.return_value = True
        mock_walk.return_value = [
            ('/path/to/dir', [], ['video1.mp4', 'video2.mkv', 'text.txt']),
            ('/path/to/dir/subdir', [], ['audio.mp3'])
        ]
        self.mock_options.file = None
        self.mock_options.directory = '/path/to/dir'
        self.mock_options.txt = None

        # Execute
        result = self.media._collect_potential_files()

        # Assert
        self.assertEqual(len(result), 3)
        # Use Path objects for comparison to handle platform-specific path separators
        expected_paths = {
            Path('/path/to/dir/video1.mp4'),
            Path('/path/to/dir/video2.mkv'),
            Path('/path/to/dir/subdir/audio.mp3')
        }
        self.assertEqual({p for p in result}, expected_paths)
        self.assertNotIn(Path('/path/to/dir/text.txt'), result)

    @patch('os.path.isfile')
    @patch('builtins.open', new_callable=mock_open, read_data='/test_files/test_video.mp4\n/test_files/test_audio.mp3\n')
    def test_collect_potential_files_txt(self, mock_file, mock_isfile):
        """Test collecting potential files from a text file."""
        # Setup
        mock_isfile.return_value = True
        self.mock_options.file = None
        self.mock_options.directory = None
        self.mock_options.txt = '/path/to/files.txt'

        # Mock Path.is_file and Path.is_dir
        with patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=False):
            # Execute
            result = self.media._collect_potential_files()

            # Assert
            self.assertEqual(len(result), 2)
            # Use Path objects for comparison to handle platform-specific path separators
            expected_paths = {
                Path('/test_files/test_video.mp4'),
                Path('/test_files/test_audio.mp3')
            }
            self.assertEqual({p for p in result}, expected_paths)

    @patch('ffmpeg.probe')
    def test_is_media_file_valid_video(self, mock_probe):
        """Test checking if a file is a valid video file."""
        # Setup
        mock_probe.return_value = {
            'streams': [{'codec_type': 'video'}]
        }
        file_path = Path('/test_files/test_video.mp4')

        # Execute
        is_valid, is_audio = self.media.is_media_file(file_path)

        # Assert
        self.assertTrue(is_valid)
        self.assertFalse(is_audio)
        mock_probe.assert_called_once_with(str(file_path))

    @patch('ffmpeg.probe')
    def test_is_media_file_valid_audio(self, mock_probe):
        """Test checking if a file is a valid audio file."""
        # Setup
        mock_probe.return_value = {
            'streams': [{'codec_type': 'audio'}]
        }
        file_path = Path('/test_files/test_audio.mp3')

        # Execute
        is_valid, is_audio = self.media.is_media_file(file_path)

        # Assert
        self.assertTrue(is_valid)
        self.assertTrue(is_audio)
        mock_probe.assert_called_once_with(str(file_path))

    @patch('ffmpeg.probe')
    def test_is_media_file_invalid(self, mock_probe):
        """Test checking if a file is an invalid media file."""
        # Setup
        mock_probe.return_value = {
            'streams': [{'codec_type': 'subtitle'}]
        }
        file_path = Path('/path/to/subtitle.srt')

        # Execute
        is_valid, is_audio = self.media.is_media_file(file_path)

        # Assert
        self.assertFalse(is_valid)
        self.assertFalse(is_audio)
        mock_probe.assert_called_once_with(str(file_path))

    @patch('ffmpeg.probe')
    def test_is_media_file_txt_file(self, mock_probe):
        """Test checking if a text file is rejected."""
        # Setup
        file_path = Path('/path/to/file.txt')

        # Execute
        is_valid, is_audio = self.media.is_media_file(file_path)

        # Assert
        self.assertFalse(is_valid)
        self.assertFalse(is_audio)
        mock_probe.assert_not_called()

    @patch('ffmpeg.probe')
    def test_is_media_file_exception(self, mock_probe):
        """Test handling exceptions when checking a file."""
        # Setup
        mock_probe.side_effect = Exception("Probe error")
        file_path = Path('/path/to/corrupted.mp4')

        # Execute
        is_valid, is_audio = self.media.is_media_file(file_path)

        # Assert
        self.assertFalse(is_valid)
        self.assertFalse(is_audio)
        mock_probe.assert_called_once_with(str(file_path))

    @patch.object(Media, '_collect_potential_files')
    @patch.object(Media, 'is_media_file')
    def test_get_media_files_success(self, mock_is_media_file, mock_collect_files):
        """Test discovering media files successfully."""
        # Setup
        file1 = Path('/test_files/test_video.mp4')
        file2 = Path('/test_files/test_audio.mp3')
        mock_collect_files.return_value = {file1, file2}
        
        # Configure is_media_file to return different values for different files
        def side_effect(file_path):
            if file_path == file1:
                return True, False  # video file
            elif file_path == file2:
                return True, True   # audio file
            return False, False
        
        mock_is_media_file.side_effect = side_effect

        # Execute
        result = self.media.get_media_files()

        # Assert
        self.assertEqual(len(result), 2)
        self.assertEqual(len(self.media._media_files), 2)
        
        # Check that MediaFile objects were created correctly
        media_files = {str(mf.path): mf.is_audio_only for mf in self.media._media_files}
        self.assertIn(str(file1), media_files)
        self.assertIn(str(file2), media_files)
        self.assertFalse(media_files[str(file1)])  # video file
        self.assertTrue(media_files[str(file2)])   # audio file

    @patch.object(Media, '_collect_potential_files')
    def test_get_media_files_no_potential_files(self, mock_collect_files):
        """Test discovering media files when no potential files are found."""
        # Setup
        mock_collect_files.return_value = set()

        # Execute and Assert
        with self.assertRaises(MediaNotFoundError):
            self.media.get_media_files()

    @patch.object(Media, '_collect_potential_files')
    @patch.object(Media, 'is_media_file')
    def test_get_media_files_no_valid_files(self, mock_is_media_file, mock_collect_files):
        """Test discovering media files when no valid files are found."""
        # Setup
        file1 = Path('/path/to/invalid1.mp4')
        file2 = Path('/path/to/invalid2.mp3')
        mock_collect_files.return_value = {file1, file2}
        mock_is_media_file.return_value = (False, False)  # All files are invalid

        # Execute and Assert
        with self.assertRaises(MediaNotFoundError):
            self.media.get_media_files()

    @patch.object(Media, '_collect_potential_files')
    @patch.object(Media, 'is_media_file')
    def test_get_media_files_validation_exception(self, mock_is_media_file, mock_collect_files):
        """Test handling exceptions during file validation."""
        # Setup
        file1 = Path('/test_files/test_video.mp4')
        file2 = Path('/path/to/error.mp3')
        mock_collect_files.return_value = {file1, file2}
        
        # Configure is_media_file to raise an exception for one file
        def side_effect(file_path):
            if file_path == file1:
                return True, False  # valid video file
            elif file_path == file2:
                raise Exception("Validation error")
            return False, False
        
        mock_is_media_file.side_effect = side_effect

        # Execute
        result = self.media.get_media_files()

        # Assert
        self.assertEqual(len(result), 1)  # Only one valid file
        self.assertEqual(len(self.media._media_files), 1)
        self.assertEqual(str(self.media._media_files[0].path), str(file1))
        self.assertFalse(self.media._media_files[0].is_audio_only)


if __name__ == '__main__':
    unittest.main()
