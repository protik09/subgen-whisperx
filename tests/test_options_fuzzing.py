import unittest
import os
import random
import string
from unittest.mock import patch

from objects.options import Options
from utils.exceptions import FolderNotFoundError, InvalidThreadCountError, InvalidPathError
from utils.constants import MODELS_AVAILABLE, WHISPER_LANGUAGE

class TestOptionsFuzzing(unittest.TestCase):
    def setUp(self):
        # Reset the singleton instance before each test
        Options._instance = None
        Options._initialized = False

    def generate_random_string(self, length=10, include_special=False):
        """Generate a random string of specified length"""
        chars = string.ascii_letters + string.digits
        if include_special:
            chars += string.punctuation + ' ' + '\t' + '\n'
        return ''.join(random.choice(chars) for _ in range(length))

    def generate_random_path(self, depth=3, include_special=False):
        """Generate a random path with specified depth"""
        parts = [self.generate_random_string(5, include_special) for _ in range(depth)]
        return os.path.join(*parts)

    def test_fuzz_file_paths(self):
        """Test various malformed file paths"""
        problematic_paths = [
            '',
            ' ',
            '.' * 1000,
            '../' * 100,
            '\\\\?\\' + 'x' * 32000,
            '\0',
            'file:///etc/passwd',
            '|',
            '<>:"/\\|?*',
            '\t\n\r',
            'COM1',
            'PRN',
            'AUX',
            'NUL',
        ]

        for path in problematic_paths:
            with self.subTest(path=path):
                # If we somehow get past the path validation, check the full instance creation
                with patch('sys.argv', ['script.py', '-f', path]):
                    with patch('os.path.isfile', return_value=False):
                        with self.assertRaises((FileNotFoundError, ValueError, OSError, InvalidPathError, SystemExit)):
                            Options()

    def test_fuzz_directory_paths(self):
        """Test various malformed directory paths"""
        problematic_paths = [
            '',
            ' ',
            '.' * 1000,
            '../' * 100,
            '\\\\?\\' + 'x' * 32000,
            '\0',
            'file:///etc/',
            '|',
            '<>:"/\\|?*',
            '\t\n\r',
        ]

        for path in problematic_paths:
            with self.subTest(path=path):
                # If we somehow get past the path validation, check the full instance creation
                with patch('sys.argv', ['script.py', '-d', path]):
                    with patch('os.path.isdir', return_value=False):
                        with self.assertRaises((FolderNotFoundError, ValueError, OSError, InvalidPathError, SystemExit)):
                            Options()

    def test_fuzz_model_sizes(self):
        """Test various invalid model sizes"""
        invalid_models = [
            '',
            ' ',
            'LARGE',  # Wrong case
            'tiny_',  # Almost correct
            'medium.en',  # Invalid combination
            'nonexistent',
            'tiny' + '\0',  # Null byte injection
            'tiny; rm -rf /',  # Command injection attempt
            'tiny\n--help',  # Argument injection attempt
            'tiny' * 100,  # Very long string
        ]

        for model in invalid_models:
            with self.subTest(model=model):
                with patch('sys.argv', ['script.py', '-m', model]):
                    with self.assertRaises((ValueError, SystemExit)):
                        Options()

    def test_fuzz_languages(self):
        """Test various invalid language codes"""
        problematic_languages = [
            '',
            ' ',
            'xyz',  # Non-existent language
            'en ' + 'x' * 1000,  # Language with trailing data
            'en\0',  # Null byte injection
            'en; rm -rf /',  # Command injection
            'en\n--help',  # Newline injection for help command
            'e' * 100,  # Very long language code
        ]

        for lang in problematic_languages:
            with self.subTest(lang=lang):
                with patch('sys.argv', ['script.py', '-l', lang]):
                    with self.assertRaises((KeyError, ValueError, SystemExit)):
                        options = Options()
                        options._sanitize_language(lang)

    def test_fuzz_compute_devices(self):
        """Test various invalid compute device specifications"""
        invalid_devices = [
            '',
            ' ',
            'GPU',
            'CPU',
            'CUDA',
            'cuda ',
            'cpu ',
            'cuda\0',
            'cuda; rm -rf /',
            'cuda\n--help',
            'c' * 100,
        ]

        for device in invalid_devices:
            with self.subTest(device=device):
                with patch('sys.argv', ['script.py', '-c', device]):
                    if device not in ['cuda', 'cpu']:
                        with self.assertRaises(SystemExit):
                            Options()

    def test_fuzz_log_levels(self):
        """Test various invalid log levels"""
        invalid_log_levels = [
            '',
            ' ',
            'debug',  # Wrong case
            'DEBUG ',
            'TRACE',  # Non-existent level
            'DEBUG\0',
            'DEBUG; rm -rf /',
            'DEBUG\n--help',
            'D' * 100,
        ]

        for level in invalid_log_levels:
            with self.subTest(level=level):
                with patch('sys.argv', ['script.py', '-log', level]):
                    if level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                        with self.assertRaises(SystemExit):
                            Options()

    def test_fuzz_num_threads(self):
        """Test various invalid thread counts"""
        problematic_threads = [
            '',
            ' ',
            '-1',
            '0',
            '9999',  # Too high
            'abc',  # Not a number
            '1.5',  # Not an integer
            '1\0',  # Null byte injection
            '1; rm -rf /',  # Command injection
            '1\n--help',  # Newline injection for help command
            '1' * 100,  # Very long thread count
        ]

        for threads in problematic_threads:
            with self.subTest(threads=threads):
                # Test the instance creation with problematic thread count
                with patch('sys.argv', ['script.py', '-n', threads]):
                    with self.assertRaises((InvalidThreadCountError, ValueError, SystemExit)):
                        Options()

    def test_fuzz_combined_args(self):
        """Test combinations of valid and invalid arguments"""
        for _ in range(50):  # Run 50 random combinations
            args = ['script.py']
            # Randomly add arguments
            if random.random() < 0.5:
                args.extend(['-f', self.generate_random_path(include_special=True)])
            if random.random() < 0.5:
                args.extend(['-d', self.generate_random_path(include_special=True)])
            if random.random() < 0.5:
                args.extend(['-m', random.choice(list(MODELS_AVAILABLE) + ['invalid_model'])])
            if random.random() < 0.5:
                args.extend(['-l', random.choice(list(WHISPER_LANGUAGE) + ['xx'])])
            if random.random() < 0.5:
                args.extend(['-c', random.choice(['cuda', 'cpu', 'gpu'])])
            if random.random() < 0.5:
                args.extend(['-log', random.choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'TRACE'])])
            if random.random() < 0.5:
                args.extend(['-n', str(random.randint(-100, 100))])

            with self.subTest(args=args):
                with patch('sys.argv', args):
                    with patch('os.path.isfile', return_value=False):
                        with patch('os.path.isdir', return_value=False):
                            try:
                                Options()
                            except (FileNotFoundError, FolderNotFoundError, ValueError, KeyError, InvalidThreadCountError, SystemExit, InvalidPathError):
                                pass  # Expected exceptions for invalid inputs
                            except Exception as e:
                                self.fail(f"Unexpected exception: {str(e)}")

    def test_fuzz_argument_injection(self):
        """Test for argument injection vulnerabilities"""
        injection_attempts = [
            '--help',  # Help flag injection
            '-h',  # Short help flag
            '--version',  # Version flag
            '--debug',  # Debug flag
            '-v',  # Verbose flag
            '--config=malicious.conf',  # Config injection
            '-f=malicious.txt',  # Parameter injection
            '--file=malicious.txt',
            '-f malicious.txt --help',  # Multiple argument injection
            '-f "malicious.txt" --help',  # Quoted injection
            '-f=malicious.txt\n--help',  # Newline injection
            '-f=malicious.txt\0--help',  # Null byte injection
        ]

        for injection in injection_attempts:
            with self.subTest(injection=injection):
                with patch('sys.argv', ['script.py', '-f', injection]):
                    with patch('os.path.isfile', return_value=False):
                        with self.assertRaises((FileNotFoundError, SystemExit, InvalidPathError)):
                            Options()

if __name__ == '__main__':
    unittest.main()