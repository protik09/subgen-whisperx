import unittest
import os
from unittest.mock import patch
from objects.options import Options
from utils.exceptions import FolderNotFoundError


class TestOptions(unittest.TestCase):
    def setUp(self):
        # Reset the singleton instance before each test
        Options._instance = None
        Options._initialized = False

    def test_singleton_pattern(self):
        opt1 = Options()
        opt2 = Options()
        self.assertEqual(id(opt1), id(opt2))

    @patch("sys.argv", ["script.py", "-f", os.path.join("..", "assets", "input.mp4")])
    def test_file_argument(self):
        with patch("os.path.isfile", return_value=True):
            opt = Options()
            self.assertEqual(opt.file, os.path.join("..", "assets", "input.mp4"))

    @patch("sys.argv", ["script.py", "-d", os.path.join("..", "assets")])
    def test_directory_argument(self):
        with patch("os.path.isdir", return_value=True):
            opt = Options()
            self.assertEqual(opt.directory, os.path.join("..", "assets"))

    @patch("sys.argv", ["script.py", "-d", "invalid_dir"])
    def test_invalid_directory(self):
        with patch("os.path.isdir", return_value=False):
            with self.assertRaises(FolderNotFoundError):
                Options()

    @patch("sys.argv", ["script.py", "-f", "invalid.mp4"])
    def test_invalid_file(self):
        with patch("os.path.isfile", return_value=False):
            with self.assertRaises(FileNotFoundError):
                Options()

    @patch("sys.argv", ["script.py", "-l", "invalid_lang"])
    def test_invalid_language(self):
        with self.assertRaises(KeyError):
            Options()

    @patch("sys.argv", ["script.py", "-m", "tiny"])
    def test_valid_model_size(self):
        with patch("os.path.isfile", return_value=True):
            opt = Options()
            self.assertEqual(opt.model_size, "tiny")

    @patch("torch.cuda")
    def test_get_device_cuda_available(self, mock_cuda):
        mock_cuda.is_available.return_value = True
        opt = Options()
        opt.device_selection = "cuda"
        self.assertEqual(opt.get_device(), "cuda")

    @patch("torch.cuda")
    def test_get_device_cuda_not_available(self, mock_cuda):
        mock_cuda.is_available.return_value = False
        opt = Options()
        opt.device_selection = "cuda"
        self.assertEqual(opt.get_device(), "cpu")

    def test_str_representation(self):
        with patch(
            "sys.argv", ["script.py", "-f", os.path.join("..", "assets", "input.mp4")]
        ):
            with patch("os.path.isfile", return_value=True):
                opt = Options()
                str_repr = str(opt)
                expected_file_path = os.path.join("..", "assets", "input.mp4")
                
                self.assertIsInstance(str_repr, str)
                # Test each expected component of the string representation
                self.assertIn(f"File: {expected_file_path}", str_repr)
                self.assertIn("Directory: None", str_repr)
                self.assertIn("Compute Device: None", str_repr)
                self.assertIn("Model Selected: None", str_repr)
                self.assertIn("Log Level: INFO", str_repr)
                self.assertIn("Media Language: None", str_repr)
                self.assertIn("Number of Threads: None", str_repr)
                self.assertIn("Input Media Paths File: None", str_repr)


if __name__ == "__main__":
    unittest.main()
