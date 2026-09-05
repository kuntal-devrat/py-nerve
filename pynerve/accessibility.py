from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from ._types import Element

if TYPE_CHECKING:
    import uiautomation as auto

logger = logging.getLogger("pynerve.accessibility")


class AccessibilityEngine:
    """Retrieves structural UI elements directly from the OS Accessibility API.

    Args:
        max_depth: Maximum depth to walk the UI tree. Deeper values find
            more elements in complex UIs (Electron, WPF) but cost more.
            Default is 6.
    """

    def __init__(self, max_depth: int = 6) -> None:
        self.platform = sys.platform
        self.active_window: auto.Control | None = None
        self.max_depth = max_depth

    def clear_active_window(self) -> None:
        """Forget a previously focused window (e.g. after it was closed)."""
        self.active_window = None

    def _active_window_valid(self) -> bool:
        """Best-effort check that the cached active window handle is still alive."""
        win = self.active_window
        if win is None:
            return False
        try:
            # Any COM access raising means the handle is stale.
            _ = win.Name
            rect = win.BoundingRectangle
            return rect is not None
        except Exception:
            self.active_window = None
            return False

    def extract_layout(self) -> list[Element]:
        """Extract visible UI elements from the desktop window hierarchy.

        Returns Elements enriched with accessibility metadata:
        control_type, is_enabled, and value fields are populated.
        Off-screen and zero-area controls are skipped.
        """
        elements: list[Element] = []

        if self.platform != "win32":
            logger.warning(
                f"Accessibility backend is currently only supported on Windows (detected {self.platform})."
            )
            return elements

        try:
            import uiautomation as auto

            roots = []
            if self._active_window_valid():
                # Prefer the focused window; fall back to desktop root only
                # when it yields nothing (avoids walking the tree twice).
                roots.append(self.active_window)
            roots.append(auto.GetRootControl())

            seen_keys: set[tuple[str, int, int]] = set()

            for root in roots:
                if not root:
                    continue
                try:
                    for control, depth in auto.WalkControl(root, maxDepth=self.max_depth):
                        if not control:
                            continue

                        name = control.Name or ""
                        value = None
                        control_type = None
                        is_enabled = None

                        try:
                            control_type = control.ControlTypeName
                        except Exception:
                            pass
                        try:
                            is_enabled = control.IsEnabled
                        except Exception:
                            pass
                        try:
                            get_vp = getattr(control, "GetValuePattern", None)
                            vp = get_vp() if callable(get_vp) else None
                            if vp is not None:
                                value = vp.Value  # type: ignore[attr-defined]
                        except Exception:
                            pass

                        # If Name is empty, attempt fallback to Value, AutomationId, or HelpText
                        display_text = name.strip()
                        if not display_text and value and value.strip():
                            display_text = value.strip()
                        if not display_text:
                            try:
                                auto_id = control.AutomationId
                                if auto_id and auto_id.strip() and not auto_id.isdigit():
                                    display_text = auto_id.strip()
                            except Exception:
                                pass
                        if not display_text:
                            try:
                                help_text = control.HelpText
                                if help_text and help_text.strip():
                                    display_text = help_text.strip()
                            except Exception:
                                pass

                        if not display_text:
                            continue

                        rect = control.BoundingRectangle
                        if not rect:
                            continue

                        width = rect.right - rect.left
                        height = rect.bottom - rect.top
                        if width <= 0 or height <= 0:
                            continue
                        # Skip off-screen / absurdly large rects (minimized or
                        # invalid controls report garbage coordinates).
                        try:
                            is_offscreen = control.IsOffscreen
                        except Exception:
                            is_offscreen = None
                        if is_offscreen is True:
                            continue
                        if abs(rect.left) > 30000 or abs(rect.top) > 30000:
                            continue
                        if width > 10000 or height > 10000:
                            continue

                        bounds = (
                            float(rect.left),
                            float(rect.top),
                            float(rect.right),
                            float(rect.bottom),
                        )
                        center_x = (bounds[0] + bounds[2]) / 2.0
                        center_y = (bounds[1] + bounds[3]) / 2.0

                        elem_key = (display_text.lower(), int(center_x), int(center_y))
                        if elem_key in seen_keys:
                            continue
                        seen_keys.add(elem_key)

                        elements.append(
                            Element(
                                text=display_text,
                                confidence=1.0,  # Accessibility queries have 100% text certainty
                                center=(center_x, center_y),
                                bounds=bounds,
                                control_type=control_type,
                                is_enabled=is_enabled,
                                value=value,
                            )
                        )
                except Exception as walk_err:
                    logger.debug("WalkControl error on root: %s", walk_err)

                # If active_window produced elements, we don't necessarily need to walk whole root
                if elements and root is self.active_window:
                    break

        except Exception as e:
            logger.error(f"Failed to extract layout via Windows UI Automation: {e}")

        logger.debug(f"Accessibility engine extracted {len(elements)} elements")
        return elements

    def wait_for_element_event(self, text: str, timeout: float = 30.0) -> Element:
        """Wait dynamically for a structural UI element to exist using optimized OS-level query loops."""
        if self.platform != "win32":
            raise NotImplementedError("Accessibility wait_for is only supported on Windows.")

        import uiautomation as auto

        from .exceptions import ElementNotFoundError

        target_lower = text.lower().strip()
        if not target_lower:
            raise ElementNotFoundError("wait_for_element_event() requires non-empty text.")

        root = self.active_window if self._active_window_valid() else None
        root = root or auto.GetRootControl()
        if not root:
            raise ElementNotFoundError("UIA Desktop Root or active window not found.")

        # Define a comparison function that matches target substring case-insensitively
        def compare_func(control: auto.Control, depth: int = 0) -> bool:
            name = control.Name
            if name and target_lower in name.lower().strip():
                rect = control.BoundingRectangle
                if rect:
                    width = rect.right - rect.left
                    height = rect.bottom - rect.top
                    if width > 0 and height > 0:
                        return True
            return False

        # Create control spec using our custom comparison function
        control_spec = auto.Control(searchFromControl=root, searchDepth=self.max_depth, Compare=compare_func)

        # WaitForExist performs optimized COM loops (extremely CPU-efficient)
        success = auto.WaitForExist(control_spec, timeout)

        if success and control_spec.Element:
            name = control_spec.Name
            rect = control_spec.BoundingRectangle
            if not rect:
                raise ElementNotFoundError(
                    f"Matched element '{text}' has no bounding rectangle."
                )
            bounds = (
                float(rect.left),
                float(rect.top),
                float(rect.right),
                float(rect.bottom),
            )
            center_x = (bounds[0] + bounds[2]) / 2.0
            center_y = (bounds[1] + bounds[3]) / 2.0
            try:
                control_type = control_spec.ControlTypeName
            except Exception:
                control_type = None

            return Element(
                text=name.strip() if name else "",
                confidence=1.0,
                center=(center_x, center_y),
                bounds=bounds,
                control_type=control_type,
            )

        raise ElementNotFoundError(
            f"Timed out waiting for element '{text}' after {timeout} seconds."
        )
