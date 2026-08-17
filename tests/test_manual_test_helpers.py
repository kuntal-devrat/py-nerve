from __future__ import annotations

from pynerve._types import Element
from scripts.manual_test import _print_element


def test_print_element_with_dict(capsys):
    el_dict = {
        "text": "Login",
        "confidence": 0.98,
        "center": [100.0, 200.0],
        "bounds": [50, 180, 150, 220],
    }
    _print_element(el_dict, index=0)
    captured = capsys.readouterr().out
    assert "[0]" in captured
    assert "'Login'" in captured
    assert "conf=0.98" in captured
    assert "center=(100,200)" in captured


def test_print_element_with_element_instance(capsys):
    el_obj = Element(
        text="Submit",
        confidence=0.99,
        center=(300.0, 400.0),
        bounds=(250.0, 380.0, 350.0, 420.0),
    )
    _print_element(el_obj)
    captured = capsys.readouterr().out
    assert "'Submit'" in captured
    assert "conf=0.99" in captured
    assert "center=(300,400)" in captured
    assert "bounds=(250, 380, 350, 420)" in captured
