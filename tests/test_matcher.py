"""Tests for the fuzzy text matcher."""

from __future__ import annotations

from pynerve._types import Element
from pynerve.matcher import filter_by_direction, find_all_matches, find_match


def _make_element(text: str, x: float = 100, y: float = 100, conf: float = 0.95) -> Element:
    return Element(
        text=text,
        confidence=conf,
        center=(x, y),
        bounds=(x - 50, y - 10, x + 50, y + 10),
    )


class TestFindMatch:
    def test_exact_match(self):
        elements = [_make_element("Settings"), _make_element("File"), _make_element("Edit")]
        result = find_match("Settings", elements)
        assert result is not None
        assert result.text == "Settings"

    def test_case_insensitive(self):
        elements = [_make_element("Settings"), _make_element("File")]
        result = find_match("settings", elements)
        assert result is not None
        assert result.text == "Settings"

    def test_partial_match(self):
        elements = [_make_element("Settings (Modified)"), _make_element("File")]
        result = find_match("Settings", elements)
        assert result is not None
        assert "Settings" in result.text

    def test_fuzzy_match_with_symbol(self):
        elements = [_make_element("Settings \u2699\ufe0f"), _make_element("File")]
        result = find_match("Settings", elements)
        assert result is not None

    def test_no_match_below_threshold(self):
        elements = [_make_element("Completely Different"), _make_element("Another")]
        result = find_match("Settings", elements, threshold=90)
        assert result is None

    def test_empty_elements(self):
        result = find_match("Settings", [])
        assert result is None

    def test_empty_text_elements_filtered(self):
        elements = [_make_element(""), _make_element("  ")]
        result = find_match("Settings", elements)
        assert result is None

    def test_contains_match(self):
        elements = [_make_element("Open File..."), _make_element("Save")]
        result = find_match("Open", elements)
        assert result is not None
        assert "Open" in result.text


class TestFindAllMatches:
    def test_returns_multiple(self):
        elements = [
            _make_element("Delete", 100, 100),
            _make_element("Delet", 200, 200),
            _make_element("Cancel", 300, 300),
        ]
        results = find_all_matches("Delete", elements)
        assert len(results) >= 2

    def test_sorted_by_score(self):
        elements = [
            _make_element("Settings (Advanced)"),
            _make_element("Settings"),
        ]
        results = find_all_matches("Settings", elements)
        if len(results) >= 2:
            assert results[0][1] >= results[1][1]

    def test_limit_parameter(self):
        elements = [_make_element("Item") for _ in range(20)]
        results = find_all_matches("Item", elements, limit=5)
        assert len(results) <= 5


class TestFilterByDirection:
    def test_right(self):
        anchor = _make_element("Label", 100, 100)
        right_el = _make_element("Button", 200, 100)
        left_el = _make_element("Button", 50, 100)
        result = filter_by_direction([right_el, left_el], anchor, "right")
        assert result is not None
        assert result.x > anchor.x

    def test_left(self):
        anchor = _make_element("Label", 200, 100)
        right_el = _make_element("Button", 300, 100)
        left_el = _make_element("Button", 100, 100)
        result = filter_by_direction([right_el, left_el], anchor, "left")
        assert result is not None
        assert result.x < anchor.x

    def test_above(self):
        anchor = _make_element("Label", 100, 200)
        above_el = _make_element("Button", 100, 100)
        below_el = _make_element("Button", 100, 300)
        result = filter_by_direction([above_el, below_el], anchor, "above")
        assert result is not None
        assert result.y < anchor.y

    def test_below(self):
        anchor = _make_element("Label", 100, 100)
        above_el = _make_element("Button", 100, 50)
        below_el = _make_element("Button", 100, 300)
        result = filter_by_direction([above_el, below_el], anchor, "below")
        assert result is not None
        assert result.y > anchor.y

    def test_no_match(self):
        anchor = _make_element("Label", 100, 100)
        result = filter_by_direction([], anchor, "right")
        assert result is None
