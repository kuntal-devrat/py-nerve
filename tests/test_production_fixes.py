"""Regression tests for production-readiness audit fixes.

Covers: agent() max_tokens, history compression bound, screenshot cap,
dry-run read-only tools, single step_delay pacing, capture cache monitor key
+ copy semantics, accessibility region filter, matcher confidence consistency,
launch validation, scroll_to validation, wait clamping, click_at validation,
explore-scroll dedup, gateway GET/total_tokens, coding_agent edit guard,
input clipboard fast-path.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import pynerve as nv
from pynerve._types import Element
from pynerve.agent import Agent, AgentConfig, _click_at_impl, _explore_scroll_impl, _wait_impl
from pynerve.core import PyNerve
from pynerve.matcher import find_match


def _el(text, x=100, y=100, conf=0.95):
    return Element(text=text, confidence=conf, center=(x, y),
                   bounds=(x - 50, y - 10, x + 50, y + 10))


# --- agent() accepts max_tokens -------------------------------------------

def test_agent_wrapper_accepts_max_tokens():
    with patch("pynerve.agent.Agent.run") as mock_run:
        mock_run.return_value = MagicMock()
        nv.run_agent("hi", max_tokens=512, dry_run=True)
        assert mock_run.called


def test_agent_config_max_tokens_sent():
    cfg = AgentConfig(max_tokens=512)
    assert cfg.max_tokens == 512


# --- history compression never exceeds max ---------------------------------

def test_compress_history_bounded():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    for i in range(30):
        msgs.append({"role": "assistant", "content": f"a{i}", "tool_calls": [{"id": i}]})
        msgs.append({"role": "tool", "tool_call_id": str(i), "content": f"r{i}"})
    out = Agent._compress_history(msgs, 14)
    assert len(out) <= 14
    assert out[0]["role"] == "system"
    assert out[1]["role"] == "user"


def test_compress_history_small_max():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    for i in range(10):
        msgs.append({"role": "tool", "tool_call_id": str(i), "content": "x"})
    out = Agent._compress_history(msgs, 5)
    assert len(out) <= 5


# --- screenshot results not truncated --------------------------------------

def test_execute_action_preserves_screenshot():
    cfg = AgentConfig(dry_run=False)
    ag = Agent(nv=MagicMock(), config=cfg, tools=[])
    big = "__SCREENSHOT_B64__" + "A" * 100_000
    ag._tool_map = {"shot": MagicMock(fn=lambda: big, name="shot")}
    # MagicMock ToolSpec needs .fn; build a simple namespace
    from pynerve.agent import ToolSpec
    ag._tool_map = {"shot": ToolSpec(name="shot", description="d",
                                     parameters={}, fn=lambda: big)}
    out = ag._execute_action({"name": "shot", "arguments": {}}, [], 1)
    assert out.startswith("__SCREENSHOT_B64__")
    assert len(out) == len(big)

    ag._tool_map = {"t": ToolSpec(name="t", description="d",
                                  parameters={}, fn=lambda: "x" * 9000)}
    out2 = ag._execute_action({"name": "t", "arguments": {}}, [], 1)
    assert len(out2) == 4000


# --- dry-run still executes read-only tools ---------------------------------

def test_dry_run_executes_observe():
    called = {}

    def fake_observe():
        called["yes"] = True
        return "screen-state"

    from pynerve.agent import ToolSpec
    cfg = AgentConfig(dry_run=True)
    ag = Agent(nv=MagicMock(), config=cfg, tools=[])
    ag._tool_map = {"observe": ToolSpec(name="observe", description="d",
                                        parameters={}, fn=fake_observe),
                    "click": ToolSpec(name="click", description="d",
                                      parameters={}, fn=lambda text: "clicked")}
    out_obs = ag._execute_action({"name": "observe", "arguments": {}}, [], 1)
    assert called.get("yes") is True
    assert out_obs == "screen-state"
    out_click = ag._execute_action({"name": "click", "arguments": {"text": "X"}}, [], 1)
    assert out_click.startswith("DRY-RUN")


# --- step_delay only before LLM call (no post-tool sleep) --------------------

def test_no_double_step_delay():
    from pynerve.agent import ToolSpec
    cfg = AgentConfig(dry_run=False, step_delay=5.0)
    ag = Agent(nv=MagicMock(), config=cfg, tools=[])
    ag._tool_map = {"t": ToolSpec(name="t", description="d", parameters={},
                                  fn=lambda: "ok")}
    with patch("pynerve.agent.time.sleep") as mock_sleep:
        ag._execute_action({"name": "t", "arguments": {}}, [], 1)
        mock_sleep.assert_not_called()


# --- wait tool clamps + reports ----------------------------------------------

def test_wait_impl_reports_clamping():
    with patch("pynerve.agent.time.sleep"):
        assert "clamped" in _wait_impl(60.0)
        assert "clamped" not in _wait_impl(1.0)


# --- click_at validation ------------------------------------------------------

def test_click_at_rejects_bad_button():
    with pytest.raises(ValueError):
        _click_at_impl(MagicMock(move_duration=0.4), 100, 100, button="middle-x")


def test_click_at_rejects_absurd_coords():
    with pytest.raises(ValueError):
        _click_at_impl(MagicMock(move_duration=0.4), 99999, 100)


# --- explore scroll keeps duplicate labels at different rows -------------------

def test_explore_scroll_keeps_duplicates():
    rows = [
        {"text": "Delete", "center": [100, 100], "confidence": 1.0, "bounds": [0, 0, 10, 10]},
        {"text": "Delete", "center": [100, 500], "confidence": 1.0, "bounds": [0, 0, 10, 10]},
    ]
    fake_nv = MagicMock()
    fake_nv.observe.return_value = rows
    fake_nv.scroll.return_value = None
    out = _explore_scroll_impl(fake_nv, direction="down", pages=1)
    assert out.count("Delete") == 2


# --- matcher exact must respect confidence -------------------------------------

def test_exact_match_below_confidence_rejected():
    els = [_el("Save", conf=0.5)]
    assert find_match("Save", els, threshold=80) is None


def test_exact_match_high_confidence_accepted():
    els = [_el("Save", conf=0.95)]
    assert find_match("Save", els, threshold=80) is not None


def test_exact_low_conf_skips_to_high_conf_duplicate():
    els = [_el("Save", x=10, conf=0.2), _el("Save", x=500, conf=0.99)]
    got = find_match("Save", els, threshold=80)
    assert got is not None
    assert got.center[0] == 500


# --- capture cache: monitor key + copy semantics --------------------------------

def test_capture_cache_monitor_key_and_copy():
    from PIL import Image
    from pynerve.capture import ScreenCapture

    cap = ScreenCapture(cache_ttl_ms=10_000)
    img1 = Image.new("RGB", (4, 4), "red")
    img2 = Image.new("RGB", (4, 4), "blue")

    with patch("pynerve.capture._native") as mock_native:
        mock_native.screenshot.side_effect = [
            _png_bytes(img1), _png_bytes(img2), _png_bytes(img1),
        ]
        a = cap.grab(region=None, monitor_index=0)
        b = cap.grab(region=None, monitor_index=1)  # different monitor -> fresh capture
        assert mock_native.screenshot.call_count == 2
        assert a.getpixel((0, 0)) != b.getpixel((0, 0))
        # Mutating returned image must not corrupt cache
        c = cap.grab(region=None, monitor_index=1)
        c.putpixel((0, 0), (0, 0, 0))
        d = cap.grab(region=None, monitor_index=1)
        assert d.getpixel((0, 0)) != (0, 0, 0)


def _png_bytes(img):
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- accessibility region filter --------------------------------------------------

def test_filter_by_region():
    els = [_el("A", x=10, y=10), _el("B", x=500, y=500)]
    out = PyNerve._filter_by_region(els, (0, 0, 100, 100))
    assert [e.text for e in out] == ["A"]
    assert PyNerve._filter_by_region(els, None) == els


def test_format_available_truncates():
    els = [_el(f"E{i}") for i in range(50)]
    out = PyNerve._format_available(els)
    assert len(out) <= 16
    assert any("more" in s for s in out)


# --- launch validation ------------------------------------------------------------

def test_launch_rejects_empty():
    p = PyNerve.__new__(PyNerve)  # avoid VisionEngine init (needs models)
    with pytest.raises(ValueError):
        PyNerve.launch(p, "   ")


# --- scroll_to validation ------------------------------------------------------------

def test_scroll_to_rejects_zero_amount():
    p = PyNerve.__new__(PyNerve)
    with pytest.raises(ValueError):
        PyNerve.scroll_to(p, "X", amount=0)


# --- capture_window/observe_window raise when focus fails -----------------------------

def test_capture_window_raises_when_focus_fails():
    p = PyNerve.__new__(PyNerve)
    p.focus_window = lambda *a, **k: False  # type: ignore[method-assign]
    from pynerve.exceptions import ElementNotFoundError
    with pytest.raises(ElementNotFoundError):
        PyNerve.capture_window(p, "NoSuchWindow123")
    with pytest.raises(ElementNotFoundError):
        PyNerve.observe_window(p, "NoSuchWindow123")


# --- gateway: GET has no body; total_tokens split ---------------------------------------

def test_gateway_call_get_has_no_body():
    import urllib.request
    from scripts import llm_gateway as gw

    seen = {}

    class FakeResp:
        status = 200

        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        seen["data"] = req.data
        return FakeResp()

    with patch.object(urllib.request, "urlopen", fake_urlopen):
        gw._call("http://x/v1", None, "GET", "/v1/models", b"{}")
    assert seen["data"] is None


def test_gateway_total_tokens_split():
    from scripts import llm_gateway as gw
    body = json.dumps({"usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()
    assert gw._total_tokens(body) == 15
    body2 = json.dumps({"usage": {"total_tokens": 42}}).encode()
    assert gw._total_tokens(body2) == 42
    assert gw._total_tokens(b"not-json") == 0


# --- coding_agent edit guard -----------------------------------------------------------------

def test_coding_edit_guard(tmp_path):
    from scripts.coding_agent import CodingAgent
    ag = CodingAgent(workspace=tmp_path)
    spec = {t.name: t for t in ag.tools}["edit_file"]
    f = tmp_path / "a.txt"
    f.write_text("foo foo", encoding="utf-8")
    out = spec.fn(path="a.txt", old_string="foo", new_string="bar")
    assert out.startswith("ERROR")
    assert f.read_text(encoding="utf-8") == "foo foo"  # unchanged


# --- input clipboard fast-path ------------------------------------------------------------------

def test_type_text_long_uses_clipboard():
    from pynerve import input as inp
    long_text = "x" * 200
    with patch.object(inp, "_native") as mock_native:
        mock_native.get_clipboard.return_value = "old"
        inp.type_text(long_text)
        mock_native.set_clipboard.assert_called()
        mock_native.key_combo.assert_called()
        # original clipboard restored
        assert mock_native.set_clipboard.call_args_list[-1][0][0] == "old"
        mock_native.type_text.assert_not_called()


def test_type_text_short_uses_char_typing():
    from pynerve import input as inp
    with patch.object(inp, "_native") as mock_native:
        inp.type_text("hi")
        mock_native.type_text.assert_called_once()
        mock_native.set_clipboard.assert_not_called()
