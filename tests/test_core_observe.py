"""Tests for observe() snapshot and the unified confidence gate."""

from __future__ import annotations

from pynerve import PyNerve
from pynerve._types import Element
from pynerve.matcher import find_all_matches, find_match


def _el(text: str, x: float = 100.0, y: float = 100.0, conf: float = 0.95) -> Element:
    return Element(
        text=text,
        confidence=conf,
        center=(x, y),
        bounds=(x - 50, y - 10, x + 50, y + 10),
    )


class TestObserve:
    def test_structure_dedup_and_order(self, monkeypatch):
        nv = PyNerve()

        def fake_extract(region=None, force_vision=False):
            return [
                _el("Zebra", x=300, y=50),
                _el("Alpha", x=10, y=20),
                _el("Alpha", x=10, y=20),  # duplicate -> deduped
            ]

        monkeypatch.setattr(nv, "_extract_layout", fake_extract)

        snapshot = nv.observe()
        assert len(snapshot) == 2
        assert snapshot[0]["text"] == "Alpha"  # sorted by y (top first)
        assert snapshot[1]["text"] == "Zebra"
        entry = snapshot[0]
        assert {"text", "confidence", "center", "bounds"}.issubset(set(entry))
        assert entry["center"] == [10.0, 20.0]
        assert len(entry["bounds"]) == 4

    def test_empty_layout(self, monkeypatch):
        nv = PyNerve()
        monkeypatch.setattr(nv, "_extract_layout", lambda region=None, force_vision=False: [])
        assert nv.observe() == []

    def test_accessibility_enriched_observe(self, monkeypatch):
        nv = PyNerve()
        enriched_el = Element(
            text="Save",
            confidence=1.0,
            center=(100.0, 100.0),
            bounds=(50.0, 90.0, 150.0, 110.0),
            control_type="Button",
            is_enabled=True,
            value="Save Document",
        )
        monkeypatch.setattr(nv, "_extract_layout", lambda region=None, force_vision=False: [enriched_el])
        obs = nv.observe()
        assert len(obs) == 1
        assert obs[0]["control_type"] == "Button"
        assert obs[0]["is_enabled"] is True
        assert obs[0]["value"] == "Save Document"

    def test_configurable_window_exclusions(self):
        nv1 = PyNerve()
        assert "visual studio code" in nv1.exclude_windows

        nv2 = PyNerve(exclude_windows=["custom_ide"])
        assert nv2.exclude_windows == ["custom_ide"]

        nv3 = PyNerve(exclude_windows=[])
        assert nv3.exclude_windows == []



class TestConfidenceUnification:
    def test_fuzzy_match_rejects_low_ocr_confidence(self):
        elements = [_el("Settings Menu", conf=0.5)]
        assert find_match("Settings", elements, threshold=80) is None

    def test_fuzzy_match_accepts_high_ocr_confidence(self):
        elements = [_el("Settings Menu", conf=0.95)]
        result = find_match("Settings", elements, threshold=80)
        assert result is not None
        assert result.text == "Settings Menu"

    def test_find_all_filters_low_confidence(self):
        elements = [
            _el("Delete", x=100, conf=0.95),
            _el("Delet", x=200, conf=0.4),
        ]
        matches = find_all_matches("Delete", elements, threshold=70)
        assert all(el.confidence >= 0.7 for el, _ in matches)
        assert len(matches) >= 1
