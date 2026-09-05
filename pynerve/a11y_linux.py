"""Linux AT-SPI accessibility backend (same interface as UIA).

Uses ``pyatspi`` when installed (``pip install dexflow[accessibility]`` on
Linux pulls it in via environment marker); otherwise returns ``[]`` so the
caller transparently falls back to vision OCR. Keeps the public contract::

    AtspiEngine(max_depth=6).extract_layout() -> list[Element]
"""

from __future__ import annotations

import logging

from ._types import Element

logger = logging.getLogger("pynerve.a11y.linux")


class AtspiEngine:
    """Structural UI walker over the AT-SPI desktop tree."""

    def __init__(self, max_depth: int = 6) -> None:
        self.max_depth = max_depth
        self.active_window = None

    def available(self) -> bool:
        """True when ``pyatspi`` can be imported."""
        try:
            import pyatspi  # noqa: F401
            return True
        except ImportError:
            return False

    def extract_layout(self) -> list[Element]:
        """Walk the AT-SPI tree depth-first; empty list when unavailable."""
        try:
            import pyatspi
        except ImportError:
            logger.debug("pyatspi not installed; AT-SPI backend unavailable.")
            return []
        try:
            registry = pyatspi.Registry.getRegistry()
            desktop = registry.getDesktop(0)
        except Exception as e:
            logger.warning("AT-SPI desktop unavailable: %s", e)
            return []

        elements: list[Element] = []
        seen: set[tuple[str, int, int]] = set()
        roots = [self.active_window] if self.active_window is not None else list(desktop)
        for root in roots:
            self._walk(root, 0, elements, seen)
            if elements and root is self.active_window:
                break
        logger.debug("AT-SPI engine extracted %d elements", len(elements))
        return elements

    def _walk(self, node, depth: int, out: list[Element], seen: set) -> None:
        import pyatspi

        if node is None or depth > self.max_depth:
            return
        try:
            name = (node.name or "").strip()
        except Exception:
            name = ""
        if name:
            try:
                extents = node.get_extents(pyatspi.DESKTOP_COORDS)
                left, top, width, height = extents.x, extents.y, extents.width, extents.height
            except Exception:
                left = top = width = height = None
            if width and height and width > 0 and height > 0 and abs(left) < 30000:
                try:
                    role = node.getRoleName()
                except Exception:
                    role = None
                try:
                    state_set = node.getState()
                    is_enabled = state_set.contains(pyatspi.STATE_ENABLED)
                except Exception:
                    is_enabled = None
                try:
                    states_off = node.getState().contains(pyatspi.STATE_OFFSCREEN)
                except Exception:
                    states_off = False
                if not states_off:
                    bounds = (float(left), float(top),
                              float(left + width), float(top + height))
                    cx, cy = (bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0
                    key = (name.lower(), int(cx), int(cy))
                    if key not in seen:
                        seen.add(key)
                        out.append(Element(text=name, confidence=1.0, center=(cx, cy),
                                           bounds=bounds, control_type=role,
                                           is_enabled=is_enabled))
        try:
            children = [node.getChildAtIndex(i) for i in range(node.childCount)]
        except Exception:
            return
        for child in children:
            self._walk(child, depth + 1, out, seen)
