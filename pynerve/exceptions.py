from __future__ import annotations


class PyNerveError(Exception):
    """Base exception for all Py-Nerve errors."""


class ElementNotFoundError(PyNerveError):
    """Raised when the target UI element cannot be found on screen."""


class VisionError(PyNerveError):
    """Raised when OCR processing fails."""


class CaptureError(PyNerveError):
    """Raised when screenshot capture fails."""


class InputError(PyNerveError):
    """Raised when mouse/keyboard input fails."""
