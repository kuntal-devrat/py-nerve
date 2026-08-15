from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
import pynerve as nv
from pynerve._types import Element
from pynerve.exceptions import ElementNotFoundError


def test_cache_region_scoping():
    p = nv.PyNerve()
    el_full = [Element("Full", 0.9, (100.0, 100.0), (50.0, 50.0, 150.0, 150.0))]
    el_sub = [Element("Sub", 0.9, (20.0, 20.0), (10.0, 10.0, 30.0, 30.0))]

    with patch.object(p.vision, "extract_layout", side_effect=[el_full, el_sub]), \
         patch("pynerve._native.capture_hash", side_effect=[111, 222, 111, 222]):
        
        res1 = p._extract_layout(region=None)
        assert res1[0].text == "Full"

        res2 = p._extract_layout(region=(10, 10, 50, 50))
        assert res2[0].text == "Sub"

        # Now sub-region request should hit its own sub-region cache, not full screen
        res3 = p._extract_layout(region=(10, 10, 50, 50))
        assert res3[0].text == "Sub"

        # And full region should still be in cache
        res4 = p._extract_layout(region=None)
        assert res4[0].text == "Full"


def test_accessibility_empty_layout_fallback_to_vision():
    p = nv.PyNerve(backend="accessibility")
    # Accessibility returns empty list (e.g. game, custom renderer)
    p.accessibility.extract_layout = MagicMock(return_value=[])

    vision_elements = [Element("Start Game", 0.99, (200.0, 200.0), (150.0, 180.0, 250.0, 220.0))]
    with patch.object(p.vision, "extract_layout", return_value=vision_elements), \
         patch("pynerve._native.capture_hash", return_value=123):
        el = p._locate("Start Game")
        assert el.text == "Start Game"
        assert el.center == (200.0, 200.0)


def test_accessibility_relative_to_fallback():
    p = nv.PyNerve(backend="accessibility")
    p.accessibility.extract_layout = MagicMock(return_value=[])

    vision_elements = [
        Element("Username:", 0.95, (100.0, 100.0), (50.0, 90.0, 150.0, 110.0)),
        Element("input_box", 0.95, (250.0, 100.0), (200.0, 90.0, 300.0, 110.0)),
    ]
    with patch.object(p.vision, "extract_layout", return_value=vision_elements), \
         patch("pynerve._native.capture_hash", return_value=123):
        el = p._locate("input_box", relative_to="Username:", direction="right")
        assert el.text == "input_box"
        assert el.center == (250.0, 100.0)


def test_glide_interference_retargeting_element_dataclass():
    p = nv.PyNerve()
    original_el = Element("Submit", 0.9, (100.0, 100.0), (80.0, 90.0, 120.0, 110.0))
    updated_el = Element("Submit", 0.9, (120.0, 140.0), (100.0, 130.0, 140.0, 150.0))

    with patch("pynerve.core.bezier_move", side_effect=[True, False]) as mock_bezier, \
         patch.object(p, "_locate", return_value=updated_el) as mock_locate, \
         patch("time.sleep"):
        res = p._glide_to_element(original_el, original_el, 0.4)
        assert res == updated_el
        mock_locate.assert_called_with("Submit")


def test_element_offset_immutability():
    p = nv.PyNerve()
    el = Element("OK", 0.9, (100.0, 100.0), (80.0, 90.0, 120.0, 110.0), control_type="Button", is_enabled=True, value=None)
    with patch.object(p, "_extract_layout", return_value=[el]):
        result = p._locate(el, offset=(20, -10))
        assert result.center == (120.0, 90.0)
        assert el.center == (100.0, 100.0)  # Original unchanged
        assert result.control_type == "Button"
        assert result.is_enabled is True
