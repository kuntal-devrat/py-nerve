from __future__ import annotations

import io
import time

from PIL import Image

from . import _native
from .exceptions import CaptureError


class ScreenCapture:
    """Cross-platform screenshot wrapper with in-memory caching."""

    def __init__(self, cache_ttl_ms: float = 100) -> None:
        self._cache_ttl = cache_ttl_ms / 1000.0
        self._last_image: Image.Image | None = None
        self._last_region: tuple[int, int, int, int] | None = None
        self._last_time: float = 0.0

    def grab(
        self,
        region: tuple[int, int, int, int] | None = None,
        monitor_index: int | None = None,
    ) -> Image.Image:
        """Capture a screenshot and return a PIL Image.

        Args:
            region: Optional (x, y, width, height) tuple. If None, captures full screen.
            monitor_index: Optional zero-based monitor index.

        Returns:
            PIL Image of the captured region.
        """
        now = time.monotonic()
        if (
            region == self._last_region
            and self._last_image is not None
            and (now - self._last_time) < self._cache_ttl
        ):
            return self._last_image

        try:
            png_bytes = _native.screenshot(region, monitor_index)
        except Exception as e:
            raise CaptureError(f"Screenshot capture failed: {e}") from e

        image = Image.open(io.BytesIO(png_bytes))
        self._last_image = image
        self._last_region = region
        self._last_time = now
        return image

    def grab_raw(
        self,
        region: tuple[int, int, int, int] | None = None,
        monitor_index: int | None = None,
    ) -> bytes:
        """Capture a screenshot and return raw PNG bytes.

        Args:
            region: Optional (x, y, width, height) tuple. If None, captures full screen.
            monitor_index: Optional zero-based monitor index.

        Returns:
            PNG image bytes.
        """
        try:
            return _native.screenshot(region, monitor_index)
        except Exception as e:
            raise CaptureError(f"Screenshot capture failed: {e}") from e

    def invalidate_cache(self) -> None:
        """Clear the cached screenshot."""
        self._last_image = None
        self._last_region = None
        self._last_time = 0.0

