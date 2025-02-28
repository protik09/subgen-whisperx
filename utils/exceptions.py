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
