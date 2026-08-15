from __future__ import annotations

import logging
import os
import sys
import time
from typing import Literal

from . import _native
from ._types import Element
from .capture import ScreenCapture
from .exceptions import ElementNotFoundError
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
        confidence: int = 80,
        move_duration: float = 0.4,
        type_interval: float = 0.05,
        exclude_windows: list[str] | None = None,
        accessibility_depth: int = 6,
        monitor_index: int | None = None,
    ) -> None:
        """Initialize the PyNerve engine.

        Args:
            backend: Engine backend ('vision' for OCR or 'accessibility' for UIA).
            lang: OCR language. Default is English.
            models_dir: Directory containing MNN model files.
            confidence: Minimum confidence threshold for element matching (0-100).
            move_duration: Default duration for mouse movement in seconds.
            type_interval: Default delay between keystrokes in seconds.
            exclude_windows: Window title substrings to skip in focus_window().
                Defaults to DEFAULT_WINDOW_EXCLUSIONS (IDEs, terminals).
                Pass [] to disable filtering.
            accessibility_depth: Maximum depth for UIA tree walks (default 6).
                Increase for deeply nested UIs (Electron, WPF).
            monitor_index: Optional zero-based monitor index for multi-monitor setups.
        """
        self.backend = backend
        self.vision = VisionEngine(lang=lang, models_dir=models_dir)
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

        if self.backend == "accessibility":
            from .accessibility import AccessibilityEngine
            self.accessibility = AccessibilityEngine(max_depth=accessibility_depth)

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

            # Enrich vision elements with Windows UI Automation tree details if available
            if sys.platform == "win32":
                try:
                    from .accessibility import AccessibilityEngine
                    uia_eng = getattr(self, "accessibility", None) or AccessibilityEngine(max_depth=4)
                    uia_elements = uia_eng.extract_layout()
                    if uia_elements:
                        enriched = []
                        for el in elements:
                            matched_uia = None
                            cx, cy = el.center
                            for u_el in uia_elements:
                                l, t, r, b = u_el.bounds
                                if l <= cx <= r and t <= cy <= b:
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

        # Accessibility backend: try extracting layout; if empty, fallback to vision
        try:
            elements = self.accessibility.extract_layout()
            if elements:
                return elements
        except Exception as e:
            logger.warning(f"UIA layout extraction failed: {e}. Falling back to Vision engine...")

        return self._extract_layout(region=region, monitor_index=mon_idx, force_vision=True)

    def _locate(self, text: str | Element, **kwargs) -> Element:
        """Locate a single element matching the target text.

        Args:
            text: The text label to find.
            **kwargs:
                relative_to: Anchor text for relative positioning.
                direction: Direction from anchor ("right", "left", "above", "below").
                offset: Optional (x, y) offset to apply to final coordinates.

        Returns:
            The matching Element.

        Raises:
            ElementNotFoundError: If no matching element is found.
        """
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
                    f"Available elements: {[e.text for e in elements]}"
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
                    f"Available elements: {[e.text for e in elements]}"
                )

            candidate_elements = [el for el, _ in matches]
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
                f"Available elements: {[e.text for e in elements]}"
            )

        # If a foreground window is active, check if any match is inside its bounds
        fg_title, fg_rect = _get_foreground_window_info()
        if fg_rect:
            fl, ft, fr, fb = fg_rect
            for m_el, _ in matches:
                if fl <= m_el.x <= fr and ft <= m_el.y <= fb:
                    return apply_offset(m_el)

        return apply_offset(matches[0][0])


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

        Returns:
            True if successful.
        """
        duration = kwargs.pop("duration", self.move_duration)
        element = self._locate(text, **kwargs)

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

        Returns:
            True if successful.
        """
        duration = kwargs.pop("duration", self.move_duration)
        element = self._locate(text, **kwargs)

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

        Returns:
            True if successful.
        """
        duration = kwargs.pop("duration", self.move_duration)
        element = self._locate(text, **kwargs)

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

        Returns:
            True if successful.
        """
        duration = kwargs.pop("duration", self.move_duration)
        element = self._locate(text, **kwargs)

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

        Returns:
            True if successful.
        """
        duration = kwargs.pop("duration", self.move_duration)
        element = self._locate(text, **kwargs)

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
                clear: If True, select all and delete before typing.

        Returns:
            True if successful.
        """
        duration = kwargs.pop("duration", self.move_duration)
        clear = kwargs.pop("clear", False)
        element = self._locate(text, **kwargs)

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

    def find(self, text: str | Element, **kwargs) -> Element:
        """Locate an element and return its position info.

        Args:
            text: The text label to find.
            **kwargs:
                relative_to: Anchor text for relative positioning.
                direction: Direction from anchor.

        Returns:
            Element with text, confidence, center, and bounds.
        """
        return self._locate(text, **kwargs)

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

    def screenshot(self, region: tuple[int, int, int, int] | None = None):
        """Capture a screenshot.

        Args:
            region: Optional (x, y, width, height) tuple.

        Returns:
            PIL Image.
        """
        return self.capture.grab(region)

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
                return self.accessibility.wait_for_element_event(target_text, timeout)
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
        import sys
        if sys.platform != "win32":
            logger.warning("focus_window is currently only supported on Windows.")
            return False

        import time
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout:
            # Method 1: Try UIA if available
            try:
                import uiautomation as auto
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
                            self.accessibility.active_window = window

                        return True
            except ImportError:
                # uiautomation not installed -> use fast ctypes Win32 EnumWindows fallback
                if _focus_window_win32_ctypes(title_substring, class_name, self.exclude_windows):
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

    def type_text(self, text: str, interval: float | None = None) -> None:
        """Type text at the current cursor position."""
        _type_text(text, interval or self.type_interval)

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
        """Capture a screenshot of a specific window by title substring."""
        self.focus_window(title_substring)
        time.sleep(0.2)
        _, fg_rect = _get_foreground_window_info()
        if fg_rect:
            l, t, r, b = fg_rect
            w = max(1, r - l)
            h = max(1, b - t)
            return self.screenshot(region=(l, t, w, h))
        return self.screenshot()

    def observe_window(self, title_substring: str) -> list[dict]:
        """Observe screen elements constrained to a specific window."""
        self.focus_window(title_substring)
        time.sleep(0.2)
        _, fg_rect = _get_foreground_window_info()
        if fg_rect:
            l, t, r, b = fg_rect
            w = max(1, r - l)
            h = max(1, b - t)
            return self.observe(region=(l, t, w, h))
        return self.observe()

    def launch(self, app: str) -> str:
        """Launch an application or open a URL using the OS app launcher (cross-platform).

        Args:
            app: An app name (e.g. ``"chrome"``), a full path to an executable, a
                 document, or a URL. Windows uses ``os.startfile`` (resolves
                 registered app names); macOS uses ``open``; Linux uses ``xdg-open``.

        Returns:
            A short confirmation string.
        """
        import platform
        import subprocess

        system = platform.system()
        if system == "Windows":
            getattr(os, "startfile")(app)  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.Popen(["open", app])
        else:
            subprocess.Popen(["xdg-open", app])
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
        """
        timeout = kwargs.pop("timeout", 15.0)
        amount = kwargs.pop("amount", -2)
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
        time.sleep(0.1)
        bezier_move(target.x, target.y, duration)
        time.sleep(0.1)
        mouse_up("left")
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
                fl, ft, fr, fb = fg_rect
                d["in_active_window"] = fl <= el.center[0] <= fr and ft <= el.center[1] <= fb
            # Include accessibility metadata when available
            if el.control_type is not None:
                d["control_type"] = el.control_type
            if el.is_enabled is not None:
                d["is_enabled"] = el.is_enabled
            if el.value is not None:
                d["value"] = el.value
            result.append(d)
        return result


