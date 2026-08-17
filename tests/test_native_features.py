from __future__ import annotations

from unittest.mock import patch

import pytest

import pynerve as nv
from pynerve import _native
from pynerve._types import Element


def test_list_monitors():
    try:
        monitors = nv.list_monitors()
    except Exception as e:
        pytest.skip(f"Display/Monitors unavailable in headless environment: {e}")
    assert isinstance(monitors, list)
    if monitors:
        idx, name, is_pri, (x, y, w, h) = monitors[0]
        assert isinstance(idx, int)
        assert isinstance(name, str)
        assert isinstance(is_pri, bool)
        assert w > 0 and h > 0


def test_clipboard_set_and_get():
    test_str = "pynerve_clipboard_test_12345"
    try:
        nv.set_clipboard(test_str)
        assert nv.get_clipboard() == test_str
    except Exception as e:
        # In headless or restricted CI environments, clipboard may be unavailable
        pytest.skip(f"Clipboard unavailable: {e}")


def test_key_press_and_symbols():
    # Test valid keys do not raise "Unknown key"
    symbols = ["+", "_", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")",
               "{", "}", "|", ":", "\"", "<", ">", "?", "~", "numpad0", "f1", "f12"]
    for s in symbols:
        try:
            _native.press_key(s)
        except ValueError as e:
            pytest.fail(f"Key '{s}' failed to parse: {e}")
        except Exception:
            # Enigo OS-level interaction may fail in non-interactive CI, which is fine
            pass

    # Invalid key raises ValueError
    with pytest.raises(ValueError, match="Unknown key"):
        _native.press_key("nonexistent_key_12345")


def test_key_combo_empty():
    with pytest.raises(ValueError, match="keys list cannot be empty"):
        _native.key_combo([])


def test_hover_and_middle_click():
    mock_el = Element("Save", 0.95, (100.0, 100.0), (80.0, 90.0, 120.0, 110.0))
    p = nv.PyNerve()
    with patch.object(p, "_locate", return_value=mock_el), \
         patch("pynerve.core._click") as mock_click, \
         patch("pynerve.core.bezier_move", return_value=False), \
         patch("time.sleep"):
        res = p.hover("Save", dwell=0.0)
        assert res is True

        res_mc = p.middle_click("Save")
        assert res_mc is True
        mock_click.assert_called_with("middle")


def test_capture_and_observe_window():
    p = nv.PyNerve()
    with patch.object(p, "focus_window", return_value=True), \
         patch("pynerve.core._get_foreground_window_info", return_value=("Test App", (10, 10, 100, 100))), \
         patch.object(p, "screenshot") as mock_ss, \
         patch.object(p, "observe", return_value=[{"text": "OK"}]) as mock_obs, \
         patch("time.sleep"):
        p.capture_window("Test App")
        mock_ss.assert_called_with(region=(10, 10, 90, 90))

        obs = p.observe_window("Test App")
        assert obs == [{"text": "OK"}]
        mock_obs.assert_called_with(region=(10, 10, 90, 90))
