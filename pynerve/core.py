from __future__ import annotations

import logging
import os
import sys
import time
from typing import Literal

from PIL import Image

from . import _native
from ._types import Element
from .capture import ScreenCapture
from .exceptions import ElementNotFoundError
from .icons import ImageMatch
from .input import bezier_move, get_position
from .input import click as _click
from .input import double_click as _double_click
from .input import key_combo as _key_combo
from .input import press_key as _press_key
from .input import type_text as _type_text
from .matcher import filter_by_direction, find_all_matches, find_match
from .vision import VisionEngine

logger = logging.getLogger("pynerve")

# Default window title substrings to skip in focus_window() to avoid
# self-focusing the IDE/terminal running the automation script.
# Users can override via PyNerve(exclude_windows=[...]) or pass [] to disable.
DEFAULT_WINDOW_EXCLUSIONS = [
    "visual studio code",
    "powershell",
    "command prompt",
    "windows terminal",
    "codebuff",
]


def _get_foreground_window_info() -> tuple[str | None, tuple[int, int, int, int] | None]:
    """Return (title, (left, top, right, bottom)) of the active foreground window on Windows."""
    import sys
    if sys.platform != "win32":
        return None, None
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None, None
        length = user32.GetWindowTextLengthW(hwnd)
        title = None
        if length > 0:
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            title = buff.value
        rect = wintypes.RECT()
        if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            w = rect.right - rect.left
            h = rect.bottom - rect.top
            if w > 50 and h > 50:
                return title, (rect.left, rect.top, rect.right, rect.bottom)
    except Exception:
        pass
    return None, None


def _focus_window_win32_ctypes(
    title_substring: str,
    class_name: str | None = None,
    exclude_windows: list[str] | None = None,
) -> bool:

    """Focus a window using pure Windows stdlib ctypes (zero external dependencies)."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    target_lower = title_substring.lower()
    exclusions = [e.lower() for e in (exclude_windows or [])]
    found_hwnd = None

    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def enum_windows_callback(hwnd: int, lparam: int) -> bool:
        nonlocal found_hwnd
        if not user32.IsWindowVisible(hwnd):
            return True

        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True

        buff = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buff, length + 1)
        title = buff.value
        title_lower = title.lower()

        # Check exclusions
        if any(excl in title_lower for excl in exclusions):
            return True

        if target_lower in title_lower:
            if class_name:
                c_buff = ctypes.create_unicode_buffer(256)
                user32.GetClassNameW(hwnd, c_buff, 256)
                if class_name.lower() not in c_buff.value.lower():
                    return True
            found_hwnd = hwnd
            return False  # found window -> stop enumeration
        return True

    cb = WNDENUMPROC(enum_windows_callback)
    user32.EnumWindows(cb, 0)

    if found_hwnd:
        # SW_RESTORE = 9, SW_SHOW = 5
        kernel32 = ctypes.windll.kernel32
        cur_thread = kernel32.GetCurrentThreadId()
        fg_hwnd = user32.GetForegroundWindow()
        fg_thread = user32.GetWindowThreadProcessId(fg_hwnd, None) if fg_hwnd else cur_thread

        if cur_thread != fg_thread:
            user32.AttachThreadInput(cur_thread, fg_thread, True)

        user32.ShowWindow(found_hwnd, 9)
        user32.BringWindowToTop(found_hwnd)
        user32.SetForegroundWindow(found_hwnd)
        user32.SetFocus(found_hwnd)

        if cur_thread != fg_thread:
            user32.AttachThreadInput(cur_thread, fg_thread, False)

        return True
    return False




class PyNerve:
    """Main orchestration engine for Py-Nerve desktop automation.

    Provides high-level API for interacting with UI elements using
    OCR-based text detection and human-like mouse movements.
    """

    def __init__(
        self,
        backend: Literal["vision", "accessibility"] = "vision",
        lang: str = "en",
        models_dir: str | None = None,
        models_base_url: str | None = None,
        confidence: int = 80,
        move_duration: float = 0.4,
        type_interval: float = 0.05,
        exclude_windows: list[str] | None = None,
        accessibility_depth: int = 6,
        monitor_index: int | None = None,
        action_timeout: float = 5.0,
        trace_path: str | None = None,
    ) -> None:
        """Initialize the PyNerve engine.

        Args:
            backend: Engine backend ('vision' for OCR or 'accessibility' for UIA).
            lang: OCR language. Default is English.
            models_dir: Directory containing MNN model files.
            models_base_url: Base URL for on-demand model-pack downloads
                (non-English languages). Falls back to ``DEXFLOW_MODELS_BASE_URL``.
            confidence: Minimum confidence threshold for element matching (0-100).
            move_duration: Default duration for mouse movement in seconds.
            type_interval: Default delay between keystrokes in seconds.
            exclude_windows: Window title substrings to skip in focus_window().
                Defaults to DEFAULT_WINDOW_EXCLUSIONS (IDEs, terminals).
                Pass [] to disable filtering.
            accessibility_depth: Maximum depth for UIA tree walks (default 6).
                Increase for deeply nested UIs (Electron, WPF).
            monitor_index: Optional zero-based monitor index for multi-monitor setups.
            action_timeout: Default auto-wait budget (seconds) for click/hover/
                type actions. Each action retries locating its target until the
                budget expires (Playwright-style auto-waiting). Pass
                ``timeout=0`` per call to disable, or ``timeout=<secs>`` to
                override. ``find()`` stays single-attempt by default.
            trace_path: Optional path for a JSONL action trace (see
                :mod:`pynerve.trace`). When set, every action is logged with
                timestamps, arguments, and outcomes.
        """
        self.backend = backend
        self.vision = VisionEngine(lang=lang, models_dir=models_dir, models_base_url=models_base_url)
        self.action_timeout = action_timeout
        self._tracer = None
        if trace_path is not None:
            from .trace import ActionTracer
            self._tracer = ActionTracer(trace_path)
            for _method in (
                "click", "double_click", "right_click", "middle_click", "hover",
                "type_into", "type_text", "press_key", "key_combo", "scroll",
                "scroll_to", "launch", "focus_window", "drag_and_drop",
                "find_image", "click_image",
            ):
                setattr(self, _method, self._tracer.wrap(getattr(self, _method), _method))
        self.capture = ScreenCapture()
        self.confidence = confidence
        self.move_duration = move_duration
        self.type_interval = type_interval
        self.monitor_index = monitor_index
        self.exclude_windows = (
            exclude_windows if exclude_windows is not None else list(DEFAULT_WINDOW_EXCLUSIONS)
        )

        # Region/monitor-scoped layout cache: screen hash gates re-running OCR
        # so static screens are never re-OCR'd.
        self._layout_cache: dict[
            tuple[tuple[int, int, int, int] | None, int | None],
            tuple[int | None, list[Element], float],
        ] = {}
        self.layout_ttl: float = 0.5

        self.accessibility_depth = accessibility_depth
        self._uia_engine = None
        if self.backend == "accessibility":
            from .accessibility import AccessibilityEngine
            self._uia_engine = AccessibilityEngine(max_depth=accessibility_depth)
            self.accessibility = self._uia_engine

    def _get_uia_engine(self, max_depth: int | None = None):
        """Return a cached AccessibilityEngine (avoids re-walking cost of new instances)."""
        from .accessibility import AccessibilityEngine

        depth = max_depth if max_depth is not None else self.accessibility_depth
        existing = getattr(self, "_uia_engine", None)
        if existing is not None and existing.max_depth >= depth:
            return existing
        eng = AccessibilityEngine(max_depth=depth)
        # Cache the deepest engine requested so far.
        if existing is None or depth > existing.max_depth:
            self._uia_engine = eng
            if self.backend == "accessibility":
                self.accessibility = eng
        return eng

    @staticmethod
    def _filter_by_region(
        elements: list[Element], region: tuple[int, int, int, int] | None
    ) -> list[Element]:
        """Constrain elements to a (x, y, w, h) region by center point."""
        if region is None:
            return elements
        rx, ry, rw, rh = region
        return [
            el
            for el in elements
            if rx <= el.center[0] <= rx + rw and ry <= el.center[1] <= ry + rh
        ]

    @staticmethod
    def _format_available(elements: list[Element], limit: int = 15) -> list[str]:
        """Truncate element lists in error messages (avoids huge LLM prompts)."""
        texts = [e.text for e in elements[:limit]]
        if len(elements) > limit:
            texts.append(f"... +{len(elements) - limit} more")
        return texts

    def invalidate_cache(self) -> None:
        """Clear cached screenshots and OCR layouts."""
        self.capture.invalidate_cache()
        self._layout_cache.clear()

    def _extract_layout(
        self,
        region: tuple[int, int, int, int] | None = None,
        monitor_index: int | None = None,
        force_vision: bool = False,
    ) -> list[Element]:
        """Get the current UI layout, reusing the cached layout if the screen didn't change.

        Vision backend: a ~1ms native screen hash gates re-running OCR, so repeated
        lookups on a static screen are nearly free. ``force_vision=True`` bypasses
        the accessibility backend (used for fallback when UIA fails).
        """
        mon_idx = monitor_index if monitor_index is not None else self.monitor_index
        cache_key = (region, mon_idx)

        if self.backend == "vision" or force_vision:
            now = time.monotonic()
            cached = self._layout_cache.get(cache_key)
            if cached is not None:
                last_hash, last_layout, last_ocr_time = cached
                screen_hash = None
                try:
                    screen_hash = _native.capture_hash(region, mon_idx)
                except Exception:
                    pass
                if (screen_hash is not None and screen_hash == last_hash) or (
                    now - last_ocr_time
                ) < self.layout_ttl:
                    return last_layout

            elements = self.vision.extract_layout(region=region, monitor_index=mon_idx)

            # Enrich vision elements with Windows UI Automation tree details if available.
            # Uses a cached engine instance (no per-call construction cost).
            if sys.platform == "win32":
                try:
                    uia_eng = self._get_uia_engine(max_depth=4)
                    uia_elements = uia_eng.extract_layout()
                    if uia_elements:
                        enriched = []
                        for el in elements:
                            matched_uia = None
                            cx, cy = el.center
                            for u_el in uia_elements:
                                left, top, right, bottom = u_el.bounds
                                if left <= cx <= right and top <= cy <= bottom:
                                    matched_uia = u_el
                                    break
                            if matched_uia:
                                enriched.append(
                                    Element(
                                        text=el.text,
                                        confidence=el.confidence,
                                        center=el.center,
                                        bounds=el.bounds,
                                        control_type=matched_uia.control_type,
                                        is_enabled=matched_uia.is_enabled,
                                        value=matched_uia.value,
                                    )
                                )
                            else:
                                enriched.append(el)
                        elements = enriched
                except Exception:
                    pass

            computed_hash = None
            try:
                computed_hash = _native.capture_hash(region, mon_idx)
            except Exception:
                pass

            self._layout_cache[cache_key] = (computed_hash, elements, time.monotonic())
            return elements

        # Accessibility backend: try extracting layout; if empty, fallback to vision.
        # Region filtering is applied by center point so observe(region=...)
        # and observe_window() behave consistently across backends.
        try:
            uia_engine = self._get_uia_engine()
            elements = uia_engine.extract_layout()
            if elements:
                return self._filter_by_region(elements, region)
        except Exception as e:
            logger.warning(f"UIA layout extraction failed: {e}. Falling back to Vision engine...")

        return self._extract_layout(region=region, monitor_index=mon_idx, force_vision=True)

    def _locate_once(self, text: str | Element, **kwargs) -> Element:
        """Single-attempt locate (no waiting). See :meth:`_locate` for auto-wait."""
        kwargs.pop("timeout", None)
        kwargs.pop("poll_interval", None)
        offset = kwargs.get("offset")
        if isinstance(text, Element):
            # Create a mutable copy of Element center/bounds if we need to apply offset
            if offset:
                ox, oy = offset
                return Element(
                    text=text.text,
                    confidence=text.confidence,
                    center=(text.center[0] + ox, text.center[1] + oy),
                    bounds=text.bounds,
                    control_type=text.control_type,
                    is_enabled=text.is_enabled,
                    value=text.value,
                )
            return text

        assert isinstance(text, str)

        relative_to = kwargs.get("relative_to")
        direction = kwargs.get("direction", "right")

        elements = self._extract_layout()
        if not elements and self.backend == "accessibility":
            elements = self._extract_layout(force_vision=True)

        if not elements:
            raise ElementNotFoundError(f"No UI elements detected on screen. Target: '{text}'")

        def apply_offset(el: Element) -> Element:
            if offset:
                ox, oy = offset
                return Element(
                    text=el.text,
                    confidence=el.confidence,
                    center=(el.center[0] + ox, el.center[1] + oy),
                    bounds=el.bounds,
                    control_type=el.control_type,
                    is_enabled=el.is_enabled,
                    value=el.value,
                )
            return el

        if relative_to:
            # Find anchor element first
            anchor = find_match(relative_to, elements, self.confidence)
            if anchor is None and self.backend == "accessibility":
                elements = self._extract_layout(force_vision=True)
                anchor = find_match(relative_to, elements, self.confidence)

            if anchor is None:
                raise ElementNotFoundError(
                    f"Anchor element not found: '{relative_to}'. "
                    f"Available elements: {self._format_available(elements)}"
                )

            # Find all matches for the target text
            matches = find_all_matches(text, elements, self.confidence, limit=20)
            if not matches and self.backend == "accessibility":
                elements = self._extract_layout(force_vision=True)
                anchor = find_match(relative_to, elements, self.confidence)
                if anchor is not None:
                    matches = find_all_matches(text, elements, self.confidence, limit=20)

            if not matches:
                raise ElementNotFoundError(
                    f"Target element not found: '{text}'. "
                    f"Available elements: {self._format_available(elements)}"
                )

            candidate_elements = [el for el, _ in matches]
            if anchor is None:  # narrowed by type-checker; guarded above
                raise ElementNotFoundError(f"Anchor element not found: '{relative_to}'.")
            result = filter_by_direction(candidate_elements, anchor, direction)
            if result is None:
                raise ElementNotFoundError(
                    f"No '{text}' found {direction} of '{relative_to}'. "
                    f"Candidates: {[e.text for e in candidate_elements]}"
                )
            return apply_offset(result)

        # Direct match: find all candidates and prioritize the one inside the active foreground window
        matches = find_all_matches(text, elements, self.confidence)
        if not matches and self.backend == "accessibility":
            logger.warning(f"Element '{text}' not found via UIA. Falling back to Vision engine...")
            elements = self._extract_layout(force_vision=True)
            matches = find_all_matches(text, elements, self.confidence)

        if not matches:
            raise ElementNotFoundError(
                f"Element not found: '{text}'. "
                f"Available elements: {self._format_available(elements)}"
            )

        # If a foreground window is active, check if any match is inside its bounds
        fg_title, fg_rect = _get_foreground_window_info()
        if fg_rect:
            fg_left, fg_top, fg_right, fg_bottom = fg_rect
            for m_el, _ in matches:
                if fg_left <= m_el.x <= fg_right and fg_top <= m_el.y <= fg_bottom:
                    return apply_offset(m_el)

        return apply_offset(matches[0][0])

    def _locate(
        self,
        text: str | Element,
        timeout: float | None = None,
        poll_interval: float = 0.5,
        **kwargs,
    ) -> Element:
        """Locate a single element, retrying until ``timeout`` expires.

        Args:
            text: The text label (or Element, returned as-is) to find.
            timeout: Seconds to keep retrying. ``None``/``0`` means a single
                immediate attempt. Action methods pass ``self.action_timeout``
                by default; ``find()`` defaults to a single attempt.
            poll_interval: Seconds between attempts.
            **kwargs: ``relative_to`` / ``direction`` / ``offset`` (see
                :meth:`_locate_once`).

        Raises:
            ElementNotFoundError: If the element never appears in budget.
        """
        if isinstance(text, Element):
            return self._locate_once(text, **kwargs)
        if timeout is None or timeout <= 0:
            return self._locate_once(text, **kwargs)
        deadline = time.monotonic() + timeout
        last_error: ElementNotFoundError | None = None
        while True:
            try:
                return self._locate_once(text, **kwargs)
            except ElementNotFoundError as e:
                last_error = e
                if time.monotonic() >= deadline:
                    break
                time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))
        assert last_error is not None  # loop always runs at least once
        raise last_error


    def _glide_to_element(self, text: str | Element, element: Element, duration: float, **kwargs) -> Element:
        """Glide to element, automatically re-verifying and re-targeting if human interference occurs."""
        was_interrupted = bezier_move(element.x, element.y, duration)
        if was_interrupted:
            label = text.text if isinstance(text, Element) else text
            logger.info("Human interference detected during glide. Re-verifying target '%s'...", label)
            self.invalidate_cache()
            time.sleep(0.15)
            try:
                element = self._locate(label, **kwargs)
                logger.info("Re-targeting '%s' at updated coordinates (%.0f, %.0f)", label, element.x, element.y)
                bezier_move(element.x, element.y, duration)
            except Exception as e:
                logger.warning("Re-locating element after interruption failed: %s", e)
        return element

    def click(self, text: str | Element, **kwargs) -> bool:
        """Move to element and left-click.

        Args:
            text: The text label to click.
            **kwargs:
                relative_to: Anchor text for relative positioning.
                direction: Direction from anchor.
                duration: Override movement duration.
                timeout: Auto-wait budget in seconds (default: action_timeout).
                    Pass 0 to disable waiting.

        Returns:
            True if successful.
        """
        duration = kwargs.pop("duration", self.move_duration)
        timeout = kwargs.pop("timeout", self.action_timeout)
        element = self._locate(text, timeout=timeout, **kwargs)

        logger.info("Clicking '%s' at (%.0f, %.0f)", text if isinstance(text, str) else text.text, element.x, element.y)
        element = self._glide_to_element(text, element, duration, **kwargs)
        _click("left")
        self.invalidate_cache()
        return True

    def double_click(self, text: str | Element, **kwargs) -> bool:
        """Move to element and double-click.

        Args:
            text: The text label to double-click.
            **kwargs:
                relative_to: Anchor text for relative positioning.
                direction: Direction from anchor.
                duration: Override movement duration.
                timeout: Auto-wait budget in seconds (default: action_timeout).
                    Pass 0 to disable waiting.

        Returns:
            True if successful.
        """
        duration = kwargs.pop("duration", self.move_duration)
        timeout = kwargs.pop("timeout", self.action_timeout)
        element = self._locate(text, timeout=timeout, **kwargs)

        logger.info("Double-clicking '%s' at (%.0f, %.0f)", text if isinstance(text, str) else text.text, element.x, element.y)
        element = self._glide_to_element(text, element, duration, **kwargs)
        _double_click()
        self.invalidate_cache()
        return True

    def right_click(self, text: str | Element, **kwargs) -> bool:
        """Move to element and right-click.

        Args:
            text: The text label to right-click.
            **kwargs:
                relative_to: Anchor text for relative positioning.
                direction: Direction from anchor.
                duration: Override movement duration.
                timeout: Auto-wait budget in seconds (default: action_timeout).
                    Pass 0 to disable waiting.

        Returns:
            True if successful.
        """
        duration = kwargs.pop("duration", self.move_duration)
        timeout = kwargs.pop("timeout", self.action_timeout)
        element = self._locate(text, timeout=timeout, **kwargs)

        logger.info("Right-clicking '%s' at (%.0f, %.0f)", text if isinstance(text, str) else text.text, element.x, element.y)
        element = self._glide_to_element(text, element, duration, **kwargs)
        _click("right")
        self.invalidate_cache()
        return True

    def hover(self, text: str | Element, dwell: float = 0.2, **kwargs) -> bool:
        """Move cursor to element and dwell there without clicking.

        Args:
            text: The text label to hover over.
            dwell: Duration to pause/dwell at target location in seconds.
            **kwargs:
                relative_to: Anchor text for relative positioning.
                direction: Direction from anchor.
                duration: Override movement duration.
                timeout: Auto-wait budget in seconds (default: action_timeout).
                    Pass 0 to disable waiting.

        Returns:
            True if successful.
        """
        duration = kwargs.pop("duration", self.move_duration)
        timeout = kwargs.pop("timeout", self.action_timeout)
        element = self._locate(text, timeout=timeout, **kwargs)

        logger.info("Hovering over '%s' at (%.0f, %.0f)", text if isinstance(text, str) else text.text, element.x, element.y)
        self._glide_to_element(text, element, duration, **kwargs)
        if dwell > 0:
            time.sleep(dwell)
        return True

    def middle_click(self, text: str | Element, **kwargs) -> bool:
        """Move to element and middle-click.

        Args:
            text: The text label to middle-click.
            **kwargs:
                relative_to: Anchor text for relative positioning.
                direction: Direction from anchor.
                duration: Override movement duration.
                timeout: Auto-wait budget in seconds (default: action_timeout).
                    Pass 0 to disable waiting.

        Returns:
            True if successful.
        """
        duration = kwargs.pop("duration", self.move_duration)
        timeout = kwargs.pop("timeout", self.action_timeout)
        element = self._locate(text, timeout=timeout, **kwargs)

        logger.info("Middle-clicking '%s' at (%.0f, %.0f)", text if isinstance(text, str) else text.text, element.x, element.y)
        element = self._glide_to_element(text, element, duration, **kwargs)
        _click("middle")
        self.invalidate_cache()
        return True

    def type_into(self, text: str | Element, content: str, **kwargs) -> bool:
        """Find an input field and type text into it.

        Args:
            text: The text label of the input field.
            content: The text to type.
            **kwargs:
                relative_to: Anchor text for relative positioning.
                direction: Direction from anchor.
                duration: Override movement duration.
                timeout: Auto-wait budget in seconds (default: action_timeout).
                    Pass 0 to disable waiting.
                clear: If True, select all and delete before typing.

        Returns:
            True if successful.
        """
        duration = kwargs.pop("duration", self.move_duration)
        clear = kwargs.pop("clear", False)
        timeout = kwargs.pop("timeout", self.action_timeout)
        element = self._locate(text, timeout=timeout, **kwargs)

        logger.info("Typing into '%s' at (%.0f, %.0f)", text if isinstance(text, str) else text.text, element.x, element.y)
        element = self._glide_to_element(text, element, duration, **kwargs)
        _click("left")

        if clear:
            _press_key("home")
            # Cmd+A on macOS, Ctrl+A elsewhere
            _key_combo(["meta", "a"] if sys.platform == "darwin" else ["ctrl", "a"])
            _press_key("backspace")
            time.sleep(0.1)

        _type_text(content, self.type_interval)
        self.invalidate_cache()
        return True

    def find(
        self, text: str | Element, timeout: float | None = None, **kwargs
    ) -> Element:
        """Locate an element and return its position info.

        Args:
            text: The text label to find.
            timeout: Auto-wait budget in seconds. Default ``None`` means a
                single immediate attempt (use e.g. ``timeout=10`` to wait).
            **kwargs:
                relative_to: Anchor text for relative positioning.
                direction: Direction from anchor.

        Returns:
            Element with text, confidence, center, and bounds.
        """
        return self._locate(text, timeout=timeout, **kwargs)

    def find_all(self, text: str, threshold: int | None = None) -> list[Element]:
        """Find all elements matching the target text.

        Args:
            text: The text to search for.
            threshold: Override default confidence threshold.

        Returns:
            List of matching Elements.
        """
        threshold = threshold or self.confidence
        elements = self._extract_layout()
        matches = find_all_matches(text, elements, threshold)
        return [el for el, _ in matches]

    def find_image(
        self,
        template: Image.Image | str,
        threshold: float = 0.9,
        region: tuple[int, int, int, int] | None = None,
        timeout: float | None = None,
    ) -> ImageMatch:
        """Locate an icon/template image on screen (non-text fallback tier).

        Args:
            template: PIL image or path to a template PNG.
            threshold: Minimum match score 0..1 (1.0 = pixel-identical).
            region: Optional (x, y, w, h) area to search (much faster).
            timeout: Retry budget in seconds. Default ``None`` = single
                attempt; pass seconds to wait for the icon to appear.

        Returns:
            ImageMatch with center, bounds, and score.

        Raises:
            ElementNotFoundError: If no match meets the threshold.
        """
        from .icons import match_template

        deadline = None if timeout is None or timeout <= 0 else time.monotonic() + timeout
        while True:
            shot = self.screenshot(region=region)
            # When a region was captured, match within it (coords already local).
            match = match_template(shot, template, threshold=threshold)
            if match is not None:
                if region is not None:
                    left, top, right, bottom = match.bounds
                    match = ImageMatch(
                        center=(match.center[0] + region[0], match.center[1] + region[1]),
                        bounds=(left + region[0], top + region[1],
                                right + region[0], bottom + region[1]),
                        score=match.score,
                    )
                return match
            if deadline is None or time.monotonic() >= deadline:
                break
            self.invalidate_cache()
            time.sleep(0.5)
        raise ElementNotFoundError(
            f"No on-screen image matched the template (threshold={threshold})."
        )

    def click_image(
        self,
        template: Image.Image | str,
        threshold: float = 0.9,
        button: str = "left",
        region: tuple[int, int, int, int] | None = None,
        timeout: float | None = None,
        **kwargs,
    ) -> bool:
        """Move to an icon/template match and click it.

        Args:
            template: PIL image or path to a template PNG.
            threshold: Minimum match score 0..1.
            button: "left" (default) or "right".
            region: Optional (x, y, w, h) area to search.
            timeout: Retry budget; defaults to ``self.action_timeout``.
            **kwargs: ``duration`` override for the glide.

        Returns:
            True if successful.
        """
        if timeout is None:
            timeout = self.action_timeout
        if button not in ("left", "right"):
            raise ValueError(f"Invalid button {button!r}: expected 'left' or 'right'.")
        duration = kwargs.pop("duration", self.move_duration)
        match = self.find_image(template, threshold=threshold, region=region, timeout=timeout)
        logger.info("Clicking image match at (%.0f, %.0f) score=%.3f",
                    match.x, match.y, match.score)
        bezier_move(match.x, match.y, duration)
        _click(button)
        self.invalidate_cache()
        return True

    def screenshot(
        self, region: tuple[int, int, int, int] | None = None
    ) -> Image.Image:
        """Capture a screenshot.

        Args:
            region: Optional (x, y, width, height) tuple.

        Returns:
            PIL Image.
        """
        img: Image.Image = self.capture.grab(region)
        return img

    def get_position(self) -> tuple[float, float]:
        """Get current mouse cursor position.

        Returns:
            Tuple of (x, y) coordinates.
        """
        return get_position()

    def wait_for(self, text: str | Element, timeout: float = 30.0) -> Element:
        """Wait dynamically for an element containing text to load.

        Uses native OS event-driven callbacks for accessibility backend,
        with automatic polling fallback to the vision backend.
        """
        if self.backend == "accessibility":
            try:
                target_text = text.text if isinstance(text, Element) else text
                found: Element = self._get_uia_engine().wait_for_element_event(
                    target_text, timeout
                )
                return found
            except ElementNotFoundError:
                raise
            except Exception as e:
                logger.warning(f"UIA wait_for failed: {e}. Falling back to Vision polling...")

        # Fallback to polling scan for vision backend. The layout cache is keyed
        # on a native screen hash, so a static screen only costs ~1ms per poll;
        # OCR runs only when the screen actually changes.
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout:
            try:
                return self.find(text)
            except ElementNotFoundError:
                time.sleep(0.5)
        raise ElementNotFoundError(
            f"Timed out waiting for element '{text}' after {timeout} seconds (Vision polling fallback)."
        )

    def focus_window(self, title_substring: str, class_name: str | None = None, timeout: float = 10.0) -> bool:
        """Find and bring an application window to the foreground, waiting up to `timeout` seconds."""
        if sys.platform != "win32":
            logger.warning("focus_window is currently only supported on Windows.")
            return False

        if not title_substring or not title_substring.strip():
            logger.warning("focus_window requires a non-empty title substring.")
            return False

        try:
            import uiautomation as auto  # type: ignore[import-not-found]
            has_uia = True
        except ImportError:
            auto = None  # type: ignore[assignment]
            has_uia = False

        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout:
            # Method 1: Try UIA if available
            try:
                if not has_uia:
                    # uiautomation not installed -> use fast ctypes Win32 EnumWindows fallback
                    if _focus_window_win32_ctypes(title_substring, class_name, self.exclude_windows):
                        return True
                    time.sleep(0.5)
                    continue
                assert auto is not None
                logger.debug("Locating window via UIA with title matching '%s'...", title_substring)
                for window in auto.GetRootControl().GetChildren():
                    title = window.Name
                    cname = window.ClassName

                    if not title:
                        continue

                    # Skip windows matching the configurable exclusion list.
                    title_lower = title.lower()
                    if any(excl in title_lower for excl in self.exclude_windows):
                        continue

                    matches_title = title_substring.lower() in title_lower
                    matches_class = class_name is None or (cname and class_name.lower() in cname.lower())

                    if matches_title and matches_class:
                        logger.info("Focusing window: '%s' (Class: %s)", title, cname)

                        if not hasattr(window, "SetActive"):
                            try:
                                win_ctrl = auto.WindowControl(searchFromControl=auto.GetRootControl(), searchDepth=1, Name=title, ClassName=cname)
                                if win_ctrl.Exists(0, 0):
                                    window = win_ctrl
                            except Exception:
                                pass

                        activated = False
                        if hasattr(window, "SetActive"):
                            try:
                                getattr(window, "SetActive")()
                                activated = True
                            except Exception as e:
                                logger.debug("window.SetActive() failed: %s", e)

                        if not activated:
                            try:
                                import ctypes
                                handle = window.NativeWindowHandle
                                if handle:
                                    ctypes.windll.user32.ShowWindow(handle, 9)
                                    ctypes.windll.user32.SetForegroundWindow(handle)
                            except Exception as e:
                                logger.warning("Win32 SetForegroundWindow fallback failed: %s", e)

                        try:
                            window.SetFocus()
                        except Exception as e:
                            logger.warning("Failed to set focus on window: %s", e)

                        if self.backend == "accessibility":
                            try:
                                self._get_uia_engine().active_window = window
                            except Exception:
                                pass

                        return True
            except Exception as e:
                logger.debug("UIA focus_window scan error: %s", e)
                if _focus_window_win32_ctypes(title_substring, class_name, self.exclude_windows):
                    return True

            time.sleep(0.5)

        logger.warning("Could not find and focus window containing '%s' after %.1f seconds.", title_substring, timeout)
        return False


    def scroll(self, amount: int, axis: str = "vertical") -> None:
        """Scroll the mouse wheel. Positive for scrolling up, negative for scrolling down.

        Args:
            amount: Number of wheel notches.
            axis: "vertical" (default) or "horizontal".
        """
        from .input import scroll as _scroll
        _scroll(amount, axis=axis)

    def press_key(self, key: str) -> None:
        """Press a single key (enter, tab, escape, etc.)."""
        _press_key(key)

    def key_combo(self, keys: list[str]) -> None:
        """Press a key combination (e.g. ['ctrl', 's'])."""
        _key_combo(keys)

    def type_text(
        self,
        text: str,
        interval: float | None = None,
        use_clipboard: bool | None = None,
    ) -> None:
        """Type text at the current cursor position.

        Long strings are pasted via clipboard automatically (see
        :func:`pynerve.input.type_text`).
        """
        _type_text(text, interval or self.type_interval, use_clipboard=use_clipboard)

    def get_clipboard(self) -> str:
        """Read text from the system clipboard."""
        from .input import get_clipboard as _get_clip
        return _get_clip()

    def set_clipboard(self, text: str) -> None:
        """Write text to the system clipboard."""
        from .input import set_clipboard as _set_clip
        _set_clip(text)

    def list_monitors(self) -> list[tuple[int, str, bool, tuple[int, int, int, int]]]:
        """List all connected monitors with their (index, name, is_primary, (x, y, width, height))."""
        from .input import list_monitors as _list_mon
        return _list_mon()

    def capture_window(self, title_substring: str) -> Image.Image:
        """Capture a screenshot of a specific window by title substring.

        Focuses the window first; raises ``ElementNotFoundError`` if the
        window cannot be focused instead of silently capturing the wrong
        window.
        """
        focused = self.focus_window(title_substring)
        if not focused:
            raise ElementNotFoundError(f"Could not focus window '{title_substring}' for capture.")
        time.sleep(0.2)
        _, fg_rect = _get_foreground_window_info()
        if fg_rect:
            left, top, right, bottom = fg_rect
            width = max(1, right - left)
            height = max(1, bottom - top)
            return self.screenshot(region=(left, top, width, height))
        return self.screenshot()

    def observe_window(self, title_substring: str) -> list[dict]:
        """Observe screen elements constrained to a specific window."""
        focused = self.focus_window(title_substring)
        if not focused:
            raise ElementNotFoundError(f"Could not focus window '{title_substring}' for observe.")
        time.sleep(0.2)
        _, fg_rect = _get_foreground_window_info()
        if fg_rect:
            left, top, right, bottom = fg_rect
            width = max(1, right - left)
            height = max(1, bottom - top)
            return self.observe(region=(left, top, width, height))
        return self.observe()

    def launch(self, app: str) -> str:
        """Launch an application or open a URL using the OS app launcher (cross-platform).

        Args:
            app: An app name (e.g. ``"chrome"``), a full path to an executable, a
                 document, or a URL. Windows uses ``os.startfile`` (resolves
                 registered app names); macOS uses ``open``; Linux uses ``xdg-open``.

        Returns:
            A short confirmation string.

        Raises:
            ValueError: If ``app`` is empty.
            OSError: If the OS launcher fails (unknown app, missing binary...).
        """
        import platform
        import subprocess

        if not app or not app.strip():
            raise ValueError("launch() requires a non-empty app name, path, or URL.")
        app = app.strip()
        system = platform.system()
        try:
            if system == "Windows":
                startfile = getattr(os, "startfile", None)
                if startfile is None:  # pragma: no cover - non-Windows interpreter
                    raise OSError("os.startfile is only available on Windows.")
                startfile(app)
            elif system == "Darwin":
                proc = subprocess.Popen(["open", app])
                if proc.poll() not in (None, 0):
                    raise OSError(f"'open {app}' exited with code {proc.poll()}.")
            else:
                proc = subprocess.Popen(["xdg-open", app])
                if proc.poll() not in (None, 0):
                    raise OSError(f"'xdg-open {app}' exited with code {proc.poll()}.")
        except (OSError, FileNotFoundError) as e:
            raise OSError(f"Failed to launch {app!r}: {e}") from e
        logger.info("Launching '%s'", app)
        time.sleep(1.0)
        self.invalidate_cache()
        return f"Launched: {app}"


    def scroll_to(self, text: str | Element, **kwargs) -> bool:
        """Scroll the mouse wheel until an element containing the target text is visible on screen.

        Args:
            text: Target text to find.
            **kwargs:
                timeout: Maximum scroll time in seconds. Default is 15.0.
                amount: Scroll increment (negative for down, positive for up). Default is -2.
                    Must be non-zero; use negative to search below the fold,
                    positive to search above.
        """
        timeout = kwargs.pop("timeout", 15.0)
        amount = kwargs.pop("amount", -2)
        if amount == 0:
            raise ValueError("scroll_to() amount must be non-zero (negative=down, positive=up).")
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout:
            try:
                # Screen hash cache handles change detection: static screens are
                # nearly free, and scrolling naturally invalidates the hash.
                self._locate(text, **kwargs)
                return True
            except ElementNotFoundError:
                self.scroll(amount)
                time.sleep(0.5)
        raise ElementNotFoundError(f"Timed out scrolling for element '{text}' after {timeout} seconds.")

    def drag_and_drop(self, source_text: str | Element, target_text: str | Element, **kwargs) -> bool:
        """Drag an element and drop it onto another element.

        Args:
            source_text: Label text of the element to drag.
            target_text: Label text of the target location to drop it on.
            **kwargs:
                duration: Override mouse movement duration.
        """
        duration = kwargs.pop("duration", self.move_duration)
        source = self._locate(source_text, **kwargs)
        target = self._locate(target_text, **kwargs)

        logger.info("Dragging '%s' at (%.0f, %.0f) and dropping onto '%s' at (%.0f, %.0f)",
                    source_text, source.x, source.y, target_text, target.x, target.y)

        from .input import bezier_move, mouse_down, mouse_up
        bezier_move(source.x, source.y, duration)
        time.sleep(0.1)
        mouse_down("left")
        try:
            time.sleep(0.1)
            bezier_move(target.x, target.y, duration)
            time.sleep(0.1)
        finally:
            # Always release the button so a mid-drag failure can't leave
            # the left button stuck down.
            try:
                mouse_up("left")
            except Exception:
                pass
        self.invalidate_cache()
        return True

    def observe(self, region: tuple[int, int, int, int] | None = None) -> list[dict]:
        """Return a deduplicated snapshot of the current UI layout as plain dicts.

        This is the primary API for AI agents: it gives an LLM the full "screen
        state" (text + confidence + center + bounds) without requiring a query
        string. Elements are deduplicated by normalized text + position and
        sorted top-to-bottom, left-to-right.

        When the accessibility backend is active, elements are enriched with
        ``control_type``, ``is_enabled``, and ``value`` fields.

        Args:
            region: Optional (x, y, width, height) region to observe.

        Returns:
            List of dicts with at least: {"text", "confidence", "center", "bounds"}.
            Accessibility-enriched dicts also include: "control_type", "is_enabled", "value".
        """
        elements = self._extract_layout(region)
        seen: set[tuple[str, int, int]] = set()
        result: list[dict] = []
        fg_title, fg_rect = _get_foreground_window_info()

        for el in sorted(elements, key=lambda e: (e.center[1], e.center[0])):
            key = (el.text.lower().strip(), int(el.center[0]), int(el.center[1]))
            if key in seen:
                continue
            seen.add(key)
            d: dict = {
                "text": el.text,
                "confidence": round(el.confidence, 3),
                "center": [round(el.center[0], 1), round(el.center[1], 1)],
                "bounds": [round(b, 1) for b in el.bounds],
            }
            if fg_rect:
                fg_left, fg_top, fg_right, fg_bottom = fg_rect
                d["in_active_window"] = (
                    fg_left <= el.center[0] <= fg_right
                    and fg_top <= el.center[1] <= fg_bottom
                )
            # Include accessibility metadata when available
            if el.control_type is not None:
                d["control_type"] = el.control_type
            if el.is_enabled is not None:
                d["is_enabled"] = el.is_enabled
            if el.value is not None:
                d["value"] = el.value
            result.append(d)
        return result


