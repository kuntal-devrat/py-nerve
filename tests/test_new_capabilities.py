"""Tests for v0.2.0 capabilities: auto-wait, image fallback, a11y dispatch,
trace/recorder/headless/pytest-plugin, lazy model packs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

import pynerve as nv
from pynerve._types import Element
from pynerve.exceptions import ElementNotFoundError


def _el(text="Save", x=100.0, y=100.0):
    return Element(text=text, confidence=0.95, center=(x, y),
                   bounds=(x - 50, y - 10, x + 50, y + 10))


# ---------------------------------------------------------------- auto-wait

class TestAutoWait:
    def test_action_retries_then_succeeds(self):
        p = nv.PyNerve()
        el = _el()
        with patch.object(p, "_locate_once",
                          side_effect=[ElementNotFoundError("x"), ElementNotFoundError("y"), el]) as m, \
             patch("pynerve.core.bezier_move", return_value=False), \
             patch("pynerve.core._click"), \
             patch("time.sleep"):
            assert p.click("Save", timeout=5) is True
            assert m.call_count == 3

    def test_timeout_expiry_raises(self):
        p = nv.PyNerve()
        with patch.object(p, "_locate_once", side_effect=ElementNotFoundError("nope")), \
             patch("time.sleep"):
            with pytest.raises(ElementNotFoundError):
                p.click("Missing", timeout=0.2)

    def test_timeout_zero_single_attempt(self):
        p = nv.PyNerve()
        with patch.object(p, "_locate_once", side_effect=ElementNotFoundError("nope")) as m, \
             patch("time.sleep") as slp:
            with pytest.raises(ElementNotFoundError):
                p.click("Missing", timeout=0)
            assert m.call_count == 1
            slp.assert_not_called()

    def test_find_default_single_attempt(self):
        p = nv.PyNerve()
        with patch.object(p, "_locate_once", side_effect=ElementNotFoundError("nope")) as m:
            with pytest.raises(ElementNotFoundError):
                p.find("Missing")
            assert m.call_count == 1

    def test_find_with_timeout_waits(self):
        p = nv.PyNerve()
        el = _el()
        with patch.object(p, "_locate_once",
                          side_effect=[ElementNotFoundError("x"), el]), \
             patch("time.sleep"):
            assert p.find("Save", timeout=5) == el

    def test_action_timeout_configurable(self):
        p = nv.PyNerve(action_timeout=0)
        assert p.action_timeout == 0
        with patch.object(p, "_locate_once", side_effect=ElementNotFoundError("nope")) as m:
            with pytest.raises(ElementNotFoundError):
                p.click("Missing")  # default timeout=0 -> single attempt
            assert m.call_count == 1


# ---------------------------------------------------------------- icons

def _screen_with_icon():
    screen = Image.new("L", (160, 120), 100)
    icon = Image.new("L", (20, 14), 220)
    # Add inner detail so the match is distinctive.
    for x in range(4, 16):
        for y in range(3, 11):
            icon.putpixel((x, y), 40)
    screen.paste(icon, (60, 50))
    return screen, icon


class TestMatchTemplate:
    def test_exact_match(self):
        from pynerve.icons import match_template
        screen, icon = _screen_with_icon()
        m = match_template(screen, icon)
        assert m is not None
        assert m.score == 1.0
        assert m.center == (70.0, 57.0)
        assert m.bounds == (60.0, 50.0, 80.0, 64.0)

    def test_no_match_below_threshold(self):
        from pynerve.icons import match_template
        screen = Image.new("L", (160, 120), 100)
        icon = Image.new("L", (20, 14), 220)
        assert match_template(screen, icon, threshold=0.9) is None

    def test_region_offsets_coordinates(self):
        from pynerve.icons import match_template
        screen, icon = _screen_with_icon()
        m = match_template(screen, icon, region=(50, 40, 60, 40))
        assert m is not None
        assert m.center == (70.0, 57.0)

    def test_region_miss(self):
        from pynerve.icons import match_template
        screen, icon = _screen_with_icon()
        assert match_template(screen, icon, region=(0, 0, 30, 30)) is None

    def test_oversize_template(self):
        from pynerve.icons import match_template
        screen = Image.new("L", (10, 10), 0)
        big = Image.new("L", (50, 50), 255)
        assert match_template(screen, big) is None


class TestFindClickImage:
    def test_find_image_success(self):
        p = nv.PyNerve()
        screen, icon = _screen_with_icon()
        with patch.object(p, "screenshot", return_value=screen.convert("RGB")):
            m = p.find_image(icon)
        assert m.score == 1.0
        assert (m.x, m.y) == (70.0, 57.0)

    def test_find_image_with_region(self):
        p = nv.PyNerve()
        screen, icon = _screen_with_icon()
        full = screen.convert("RGB")

        def fake_shot(region=None):
            if region is None:
                return full
            x, y, w, h = region
            return full.crop((x, y, x + w, y + h))

        with patch.object(p, "screenshot", side_effect=fake_shot) as shot:
            m = p.find_image(icon, region=(50, 40, 60, 40))
            shot.assert_called_with(region=(50, 40, 60, 40))
        assert (m.x, m.y) == (70.0, 57.0)

    def test_find_image_miss_raises(self):
        p = nv.PyNerve()
        screen = Image.new("RGB", (160, 120), (100, 100, 100))
        icon = Image.new("L", (20, 14), 220)
        with patch.object(p, "screenshot", return_value=screen):
            with pytest.raises(ElementNotFoundError):
                p.find_image(icon)

    def test_click_image(self):
        p = nv.PyNerve()
        screen, icon = _screen_with_icon()
        with patch.object(p, "screenshot", return_value=screen.convert("RGB")), \
             patch("pynerve.core.bezier_move", return_value=False) as mv, \
             patch("pynerve.core._click") as cl:
            assert p.click_image(icon, timeout=0) is True
            mv.assert_called_once()
            assert mv.call_args[0][:2] == (70.0, 57.0)
            cl.assert_called_once_with("left")

    def test_click_image_bad_button(self):
        p = nv.PyNerve()
        with pytest.raises(ValueError):
            p.click_image(Image.new("L", (5, 5)), button="middle")


# ---------------------------------------------------------------- a11y dispatch

class TestA11yDispatch:
    def test_atspi_without_pyatspi_returns_empty(self):
        from pynerve.accessibility import AccessibilityEngine
        eng = AccessibilityEngine(force_backend="atspi")
        with patch.dict(sys.modules, {"pyatspi": None}):
            assert eng.extract_layout() == []

    def test_axui_on_windows_returns_empty(self):
        from pynerve.accessibility import AccessibilityEngine
        eng = AccessibilityEngine(force_backend="axui")
        # find_library fails on win32 -> unavailable -> []
        assert eng.extract_layout() == []

    def test_backend_name_selection(self):
        from pynerve.accessibility import AccessibilityEngine
        assert AccessibilityEngine(force_backend="uia")._backend_name() == "uia"
        assert AccessibilityEngine(force_backend="atspi")._backend_name() == "atspi"

    def test_wait_for_poll_fallback(self):
        from pynerve.accessibility import AccessibilityEngine
        eng = AccessibilityEngine(force_backend="atspi")
        with patch.object(eng, "extract_layout", return_value=[_el("Target")]), \
             patch("time.sleep"):
            el = eng.wait_for_element_event("target", timeout=5)
            assert el.text == "Target"

    def test_wait_for_poll_timeout(self):
        from pynerve.accessibility import AccessibilityEngine
        eng = AccessibilityEngine(force_backend="atspi")
        with patch.object(eng, "extract_layout", return_value=[]), \
             patch("time.sleep"):
            with pytest.raises(ElementNotFoundError):
                eng.wait_for_element_event("missing", timeout=0.1)


# ---------------------------------------------------------------- trace

class TestTracer:
    def test_log_and_read(self, tmp_path):
        from pynerve.trace import ActionTracer
        t = ActionTracer(tmp_path / "run.jsonl")
        t.log("click", {"text": "Save"}, ok=True, elapsed_ms=12.0)
        t.log("click", {"text": "X"}, ok=False, error="ElementNotFoundError: X")
        events = t.read()
        assert [e["seq"] for e in events] == [1, 2]
        assert events[0]["ok"] is True
        assert "ElementNotFound" in events[1]["error"]

    def test_wrap_success_and_failure(self, tmp_path):
        from pynerve.trace import ActionTracer
        t = ActionTracer(tmp_path / "run.jsonl")

        def ok_fn(a, b=1):
            return a + b

        def bad_fn():
            raise ValueError("boom")

        assert t.wrap(ok_fn)(2, b=3) == 5
        with pytest.raises(ValueError):
            t.wrap(bad_fn)()
        events = t.read()
        assert events[0]["ok"] is True and events[0]["args"] == {"arg0": "2", "b": "3"}
        assert events[1]["ok"] is False and "boom" in events[1]["error"]

    def test_render_html(self, tmp_path):
        from pynerve.trace import ActionTracer, render_html, render_html_file
        t = ActionTracer(tmp_path / "run.jsonl")
        t.log("click", {"text": "Save"}, ok=True, elapsed_ms=100)
        t.log("click", {"text": "X"}, ok=False, error="nope", elapsed_ms=50)
        page = render_html(t.read())
        assert "click" in page and "fail" in page
        out = render_html_file(tmp_path / "run.jsonl")
        assert out.exists() and out.suffix == ".html"

    def test_pynerve_trace_wiring(self, tmp_path):
        p = nv.PyNerve(trace_path=str(tmp_path / "t.jsonl"))
        assert hasattr(p.click, "__wrapped__")
        assert p._tracer is not None


# ---------------------------------------------------------------- headless

class TestHeadless:
    def test_no_virtual_display_on_windows(self):
        from pynerve.headless import needs_virtual_display
        with patch.object(sys, "platform", "win32"), \
             patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("DISPLAY", None)
            os.environ.pop("WAYLAND_DISPLAY", None)
            assert needs_virtual_display() is False

    def test_headless_passthrough(self):
        import importlib
        headless_mod = importlib.import_module("pynerve.headless")
        from pynerve.headless import headless
        with patch.object(headless_mod, "needs_virtual_display", return_value=False):
            with headless() as display:
                assert display is None or isinstance(display, str)

    def test_headless_linux_lifecycle(self):
        import importlib
        import os
        headless_mod = importlib.import_module("pynerve.headless")
        from pynerve.headless import headless
        proc = MagicMock()
        proc.poll.return_value = None
        with patch.object(headless_mod, "needs_virtual_display", return_value=True), \
             patch.object(headless_mod, "xvfb_available", return_value=True), \
             patch.object(headless_mod.subprocess, "Popen", return_value=proc) as popen, \
             patch.dict(os.environ, {}, clear=True), \
             patch("time.sleep"):
            with headless(display=":98") as display:
                assert display == ":98"
                assert os.environ["DISPLAY"] == ":98"
            popen.assert_called_once()
            proc.terminate.assert_called_once()
            assert "DISPLAY" not in os.environ


# ---------------------------------------------------------------- model packs

class TestModelPacks:
    def test_pack_filenames(self):
        from pynerve.model_packs import pack_filenames
        assert "en_PP-OCRv5_mobile_rec_infer.mnn" in pack_filenames("en")
        assert "PP-OCRv5_mobile_rec.mnn" in pack_filenames("fr")

    def test_pack_complete(self, tmp_path):
        from pynerve.model_packs import pack_complete
        ok, missing = pack_complete("en", tmp_path)
        assert ok is False and len(missing) == 3
        for name in missing:
            (tmp_path / name).write_bytes(b"x")
        ok, _ = pack_complete("en", tmp_path)
        assert ok is True

    def test_ensure_no_base_url(self, tmp_path):
        from pynerve.model_packs import ensure_lang_pack
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("DEXFLOW_MODELS_BASE_URL", None)
            assert ensure_lang_pack("fr", tmp_path, base_url=None) is False

    def test_ensure_downloads_missing(self, tmp_path):
        import io
        import urllib.request
        from pynerve.model_packs import ensure_lang_pack

        payloads = {name: b"data-" + name.encode() for name in
                    ["PP-OCRv5_mobile_det.mnn", "PP-OCRv5_mobile_rec.mnn", "ppocr_keys_v5.txt"]}

        class FakeResp:
            def __init__(self, data):
                self._io = io.BytesIO(data)

            def read(self, n=-1):
                return self._io.read(n)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=None):
            name = req.full_url.rsplit("/", 1)[-1]
            return FakeResp(payloads[name])

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            assert ensure_lang_pack("fr", tmp_path, base_url="https://m.test") is True
        for name, data in payloads.items():
            assert (tmp_path / name).read_bytes() == data

    def test_download_checksum_mismatch(self, tmp_path):
        import io
        import urllib.request
        from pynerve import model_packs
        from pynerve.model_packs import _download
        model_packs.SHASUMS["x.mnn"] = "0" * 64
        try:
            class FakeResp:
                def read(self, n=-1):
                    return b"" if getattr(self, "_done", False) else setattr(self, "_done", True) or b"zzz"

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

            with patch.object(urllib.request, "urlopen", return_value=FakeResp()):
                with pytest.raises(ValueError):
                    _download("https://m.test/x.mnn", tmp_path / "x.mnn", 10)
        finally:
            del model_packs.SHASUMS["x.mnn"]


# ---------------------------------------------------------------- recorder

class FakeNv:
    def __init__(self):
        self.log = []

    def click(self, text, **kw):
        self.log.append(("click", text))
        return True

    def find(self, text):
        return f"found:{text}"


class TestRecorder:
    def test_records_and_restores(self):
        from pynerve.recorder import Recorder
        fake = FakeNv()
        rec = Recorder(fake)
        with rec:
            fake.click("Save")
            assert fake.find("X") == "found:X"
        assert len(rec.calls) == 2
        assert rec.calls[0]["action"] == "click"
        # methods restored
        assert not hasattr(fake.click, "__wrapped__")

    def test_replay_mode_skips_mutation(self):
        from pynerve.recorder import Recorder
        fake = FakeNv()
        rec = Recorder(fake, replay=True)
        with rec:
            assert fake.click("Save") is True
            assert fake.find("X") == "found:X"  # read-only still executes
        assert fake.log == []  # click not executed
        assert len(rec.calls) == 2

    def test_export_script(self, tmp_path):
        from pynerve.recorder import Recorder
        fake = FakeNv()
        rec = Recorder(fake)
        with rec:
            fake.click("Save")
        out = rec.export_script(tmp_path / "flow.py")
        src = out.read_text(encoding="utf-8")
        assert 'nv.click' in src and "'Save'" in src
        assert 'if __name__ == "__main__":' in src


# ---------------------------------------------------------------- plugin + misc

class TestPlugin:
    def test_plugin_imports_and_options(self):
        from pynerve import pytest_plugin
        assert hasattr(pytest_plugin, "nv")
        assert hasattr(pytest_plugin, "dexflow")
        assert hasattr(pytest_plugin, "pytest_runtest_makereport")

    def test_hook_noop_without_fixture(self):
        from pynerve import pytest_plugin
        item = MagicMock()
        item.funcargs = {}
        call = MagicMock()
        gen = pytest_plugin.pytest_runtest_makereport(item, call)
        # hookwrapper: prime to the yield
        next(gen)
        try:
            gen.send(MagicMock(get_result=lambda: MagicMock(when="call", passed=False)))
        except StopIteration:
            pass  # no fixture -> clean return


class TestAgentImageTools:
    def test_tools_registered(self):
        from pynerve.agent import build_tools
        names = {t.name for t in build_tools(nv=MagicMock())}
        assert {"find_image", "click_image"} <= names

    def test_find_image_is_read_only(self):
        from pynerve.agent import Agent
        assert "find_image" in Agent.READ_ONLY_TOOLS
        assert "click_image" not in Agent.READ_ONLY_TOOLS


class TestVersionConsistency:
    def test_versions_match(self):
        import re
        root = Path(__file__).resolve().parent.parent
        versions = set()
        for rel in ["pyproject.toml", "Cargo.toml", "pynerve/__init__.py", "dexflow/__init__.py"]:
            text = (root / rel).read_text(encoding="utf-8")
            m = re.search(r'version__ = "([^"]+)"|\nversion = "([^"]+)"', text)
            assert m, rel
            versions.add(m.group(1) or m.group(2))
        assert versions == {"0.2.0"}, versions
