"""Headless-display helper for CI (Linux Xvfb).

Desktop automation needs a display server; Linux CI runners have none.
:func:`headless` starts ``Xvfb`` on demand and points ``DISPLAY`` at it::

    from pynerve.headless import headless

    with headless():
        nv.click("Login")

No-op on Windows/macOS and when ``DISPLAY`` is already set. Stdlib only;
requires the ``Xvfb`` binary on Linux (``apt install xvfb``).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger("pynerve.headless")


def needs_virtual_display() -> bool:
    """True on Linux without a ``DISPLAY`` (and without Wayland)."""
    return (
        sys.platform.startswith("linux")
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")
    )


def xvfb_available() -> bool:
    """True when the ``Xvfb`` binary is on PATH."""
    return shutil.which("Xvfb") is not None


@contextmanager
def headless(
    width: int = 1920,
    height: int = 1080,
    depth: int = 24,
    display: str = ":99",
) -> Iterator[str | None]:
    """Ensure a display exists; yield the active ``DISPLAY`` value.

    Starts ``Xvfb`` only when :func:`needs_virtual_display` is true and the
    binary exists. Otherwise yields the current ``DISPLAY`` unchanged (or
    ``None`` on Windows/macOS).
    """
    previous = os.environ.get("DISPLAY")
    proc = None
    if needs_virtual_display():
        if not xvfb_available():
            logger.warning("No DISPLAY and Xvfb not installed; captures will fail. "
                           "Install with: apt install xvfb")
            yield previous
            return
        cmd = ["Xvfb", display, "-screen", "0", f"{width}x{height}x{depth}"]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.environ["DISPLAY"] = display
        # Wait for the socket (up to ~5s).
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                logger.warning("Xvfb exited early with code %s.", proc.returncode)
                break
            time.sleep(0.1)
        logger.info("Started Xvfb on %s (%dx%dx%d).", display, width, height, depth)
    try:
        yield os.environ.get("DISPLAY", previous)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            if previous is None:
                os.environ.pop("DISPLAY", None)
            else:
                os.environ["DISPLAY"] = previous
            logger.info("Stopped Xvfb on %s.", display)
