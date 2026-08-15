"""Tests for exceptions."""

from __future__ import annotations

import pytest

from pynerve.exceptions import (
    CaptureError,
    ElementNotFoundError,
    InputError,
    PyNerveError,
    VisionError,
)


class TestExceptions:
    def test_hierarchy(self):
        assert issubclass(PyNerveError, Exception)
        assert issubclass(ElementNotFoundError, PyNerveError)
        assert issubclass(VisionError, PyNerveError)
        assert issubclass(CaptureError, PyNerveError)
        assert issubclass(InputError, PyNerveError)

    def test_message(self):
        err = ElementNotFoundError("not found")
        assert str(err) == "not found"

    def test_catch_with_base(self):
        with pytest.raises(PyNerveError):
            raise ElementNotFoundError("test")
