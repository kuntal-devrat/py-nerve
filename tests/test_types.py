"""Tests for the Element dataclass."""

from __future__ import annotations

import pytest

from pynerve._types import Element


class TestElement:
    def test_creation(self):
        el = Element(
            text="Hello",
            confidence=0.95,
            center=(100.0, 200.0),
            bounds=(50.0, 190.0, 150.0, 210.0),
        )
        assert el.text == "Hello"
        assert el.confidence == 0.95
        assert el.center == (100.0, 200.0)

    def test_properties(self):
        el = Element(
            text="Test",
            confidence=0.8,
            center=(100.0, 200.0),
            bounds=(50.0, 190.0, 150.0, 210.0),
        )
        assert el.x == 100.0
        assert el.y == 200.0
        assert el.width == 100.0
        assert el.height == 20.0

    def test_frozen(self):
        el = Element(text="X", confidence=1.0, center=(0, 0), bounds=(0, 0, 0, 0))
        with pytest.raises(AttributeError):
            el.text = "Y"  # type: ignore

    def test_hashable(self):
        el1 = Element(text="A", confidence=1.0, center=(0, 0), bounds=(0, 0, 0, 0))
        el2 = Element(text="A", confidence=1.0, center=(0, 0), bounds=(0, 0, 0, 0))
        assert el1 == el2
        assert hash(el1) == hash(el2)

    def test_accessibility_fields(self):
        el = Element(
            text="Submit",
            confidence=1.0,
            center=(150.0, 300.0),
            bounds=(100.0, 280.0, 200.0, 320.0),
            control_type="Button",
            is_enabled=True,
            value="Click Me",
        )
        assert el.control_type == "Button"
        assert el.is_enabled is True
        assert el.value == "Click Me"

