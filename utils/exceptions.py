"""
Custom Exception classes for better and clearer error handling.
"""


class FolderNotFoundError(Exception):
    """
    Exception raised when the folder is not found.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.message = kwargs.get("message")

    def __str__(self):
        return f"{self.message}"
    
class MediaNotFoundError(Exception):
    """
    Exception raised when the folder is not found.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.message = kwargs.get("message")

    def __str__(self):
        return f"{self.message}"

class InvalidThreadCountError(Exception):
    """
    Exception raised when an invalid thread count is specified.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message

class InvalidPathError(Exception):
    """
    Exception raised when a path is invalid or potentially malicious.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message
