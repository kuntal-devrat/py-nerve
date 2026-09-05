"""Pytest plugin: fixtures + failure artifacts for desktop automation.

Enabled via entry point (``pip install dexflow`` is enough)::

    def test_login(nv):
        nv.click("Login", timeout=10)
        assert nv.find("Welcome", timeout=10)

On failure the plugin saves ``<test>_screenshot.png``,
``<test>_observe.json`` and the JSONL ``<test>_trace.jsonl`` into
``--dexflow-artifacts`` (default ``target/pytest-artifacts``). All capture
is best-effort — headless runners without a display log a warning instead
of failing the test.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

import pytest

logger = logging.getLogger("pynerve.pytest")


def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("dexflow", "desktop automation")
    group.addoption(
        "--dexflow-artifacts",
        default="target/pytest-artifacts",
        help="Directory for failure screenshots/traces (default: target/pytest-artifacts).",
    )
    group.addoption(
        "--dexflow-backend",
        default="vision",
        help="PyNerve backend for the nv fixture (default: vision).",
    )


def _artifacts_dir(request: Any) -> Path:
    out = Path(request.config.getoption("--dexflow-artifacts"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def _make_nv(request: Any) -> Any:
    from .core import PyNerve

    backend = request.config.getoption("--dexflow-backend")
    trace_path = str(_artifacts_dir(request) / f"{request.node.name}_trace.jsonl")
    return PyNerve(backend=backend, trace_path=trace_path)


@pytest.fixture
def nv(request: Any) -> Iterator[Any]:
    """Function-scoped traced :class:`PyNerve` instance."""
    instance = _make_nv(request)
    yield instance
    instance.invalidate_cache()


@pytest.fixture
def dexflow(request: Any) -> Iterator[Any]:
    """Alias of the :func:`nv` fixture."""
    instance = _make_nv(request)
    yield instance
    instance.invalidate_cache()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: Any, call: Any) -> Iterator[Any]:
    outcome: Any = yield
    report = outcome.get_result()
    if report.when != "call" or report.passed:
        return
    instance = item.funcargs.get("nv") or item.funcargs.get("dexflow")
    if instance is None:
        return
    try:
        artifacts = Path(item.config.getoption("--dexflow-artifacts"))
        artifacts.mkdir(parents=True, exist_ok=True)
        stem = item.name.replace("/", "_").replace("\\", "_")
        try:
            shot = instance.screenshot()
            shot.save(artifacts / f"{stem}_screenshot.png")
        except Exception as e:
            logger.warning("Could not capture failure screenshot: %s", e)
        try:
            layout = instance.observe()
            (artifacts / f"{stem}_observe.json").write_text(
                json.dumps(layout[:200], indent=1), encoding="utf-8")
        except Exception as e:
            logger.warning("Could not capture failure layout: %s", e)
        try:
            from .trace import render_html_file
            trace = artifacts / f"{stem}_trace.jsonl"
            if trace.exists():
                render_html_file(trace)
        except Exception as e:
            logger.warning("Could not render failure trace: %s", e)
    except Exception as e:
        logger.warning("Dexflow artifact hook failed: %s", e)
