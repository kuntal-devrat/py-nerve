"""macOS AXUIElement accessibility backend (same interface as UIA).

System Accessibility API via ``ctypes`` — no third-party dependency.
The process must be trusted for accessibility
(System Settings → Privacy & Security → Accessibility); otherwise
:func:`AxuiEngine.extract_layout` returns ``[]`` and callers fall back to
vision OCR::

    AxuiEngine(max_depth=6).extract_layout() -> list[Element]
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging

from ._types import Element

logger = logging.getLogger("pynerve.a11y.macos")

_kAXUTF8 = 0x08000100  # kCFStringEncodingUTF8

# AX roles that never carry clickable meaning; skipped to reduce noise.
_SKIP_ROLES = frozenset({"AXUnknown", "AXSplitter", "AXScrollArea"})


class _AX:
    """Lazily-bound ApplicationServices + CoreFoundation handles."""

    def __init__(self) -> None:
        app_path = ctypes.util.find_library("ApplicationServices")
        cf_path = ctypes.util.find_library("CoreFoundation")
        if not app_path or not cf_path:
            raise OSError("ApplicationServices/CoreFoundation not found.")
        self.app = ctypes.CDLL(app_path, use_errno=True)
        self.cf = ctypes.CDLL(cf_path, use_errno=True)
        self._bind()

    def _bind(self) -> None:
        app, cf = self.app, self.cf
        app.AXUIElementCreateSystemWide.restype = ctypes.c_void_p
        app.AXUIElementCopyAttributeValue.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        app.AXUIElementCopyAttributeValue.restype = ctypes.c_int32
        app.AXIsProcessTrusted.restype = ctypes.c_bool
        try:
            app.AXIsProcessTrustedWithOptions.argtypes = [ctypes.c_void_p]
            app.AXIsProcessTrustedWithOptions.restype = ctypes.c_bool
        except AttributeError:
            pass
        app.AXValueGetValue.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
        app.AXValueGetValue.restype = ctypes.c_bool
        cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
        cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
        cf.CFStringGetCString.restype = ctypes.c_bool
        cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
        cf.CFArrayGetCount.restype = ctypes.c_long
        cf.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]
        cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
        cf.CFGetTypeID.argtypes = [ctypes.c_void_p]
        cf.CFGetTypeID.restype = ctypes.c_ulong
        cf.CFStringGetTypeID.restype = ctypes.c_ulong
        cf.CFBooleanGetValue.argtypes = [ctypes.c_void_p]
        cf.CFBooleanGetValue.restype = ctypes.c_bool
        cf.CFNumberGetValue.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
        cf.CFNumberGetValue.restype = ctypes.c_bool
        cf.CFRelease.argtypes = [ctypes.c_void_p]
        cf.CFRelease.restype = None

    def const(self, name: str) -> ctypes.c_void_p:
        """Read an exported CFStringRef constant (e.g. kAXRoleAttribute)."""
        return ctypes.c_void_p.in_dll(self.app, name)


class AxuiEngine:
    """Structural UI walker over the macOS accessibility tree."""

    def __init__(self, max_depth: int = 6) -> None:
        self.max_depth = max_depth
        self.active_window = None  # reserved: PID to scope the walk
        self._ax: _AX | None = None

    # -- public contract ------------------------------------------------------

    def available(self) -> bool:
        try:
            self._ensure()
        except Exception:
            return False
        return self._ax is not None and self._ax.app.AXIsProcessTrusted()

    def extract_layout(self) -> list[Element]:
        try:
            self._ensure()
        except Exception as e:
            logger.debug("AXUI unavailable: %s", e)
            return []
        assert self._ax is not None
        if not self._ax.app.AXIsProcessTrusted():
            logger.debug("Process not trusted for accessibility; AXUI unavailable.")
            return []
        try:
            system = self._ax.app.AXUIElementCreateSystemWide()
            app_ref = self._attr(system, "kAXFocusedApplicationAttribute")
            self._ax.cf.CFRelease(system)
            if not app_ref:
                return []
            elements: list[Element] = []
            seen: set[tuple[str, int, int]] = set()
            for win in self._array(app_ref, "kAXWindowsAttribute"):
                self._walk(win, 0, elements, seen)
                self._ax.cf.CFRelease(win)
            self._ax.cf.CFRelease(app_ref)
            logger.debug("AXUI engine extracted %d elements", len(elements))
            return elements
        except Exception as e:
            logger.debug("AXUI walk failed: %s", e)
            return []

    # -- internals --------------------------------------------------------------

    def _ensure(self) -> None:
        if self._ax is None:
            self._ax = _AX()

    def _attr(self, element: int, const_name: str) -> int | None:
        """Copy one attribute; returns retained CFTypeRef address or None."""
        assert self._ax is not None
        out = ctypes.c_void_p()
        err = self._ax.app.AXUIElementCopyAttributeValue(
            ctypes.c_void_p(element),
            self._ax.const(const_name),
            ctypes.byref(out),
        )
        if err != 0 or not out.value:
            return None
        return out.value

    def _array(self, element: int, const_name: str) -> list[int]:
        """Copy a CFArray attribute as retained addresses (caller releases)."""
        assert self._ax is not None
        arr = self._attr(element, const_name)
        if not arr:
            return []
        try:
            count = self._ax.cf.CFArrayGetCount(ctypes.c_void_p(arr))
            return [self._ax.cf.CFArrayGetValueAtIndex(ctypes.c_void_p(arr), i)
                    for i in range(count)]
        finally:
            self._ax.cf.CFRelease(arr)

    def _string(self, ref: int) -> str | None:
        assert self._ax is not None
        if self._ax.cf.CFGetTypeID(ctypes.c_void_p(ref)) != self._ax.cf.CFStringGetTypeID():
            return None
        buf = ctypes.create_string_buffer(1024)
        ok = self._ax.cf.CFStringGetCString(ctypes.c_void_p(ref), buf, 1024, _kAXUTF8)
        return buf.value.decode("utf-8", errors="replace") if ok else None

    def _walk(self, element: int, depth: int, out: list[Element], seen: set) -> None:
        assert self._ax is not None
        if depth > self.max_depth:
            return
        try:
            role_ref = self._attr(element, "kAXRoleAttribute")
            role = self._string(role_ref) if role_ref else None
            if role_ref:
                self._ax.cf.CFRelease(role_ref)
            if role in _SKIP_ROLES:
                return

            text = None
            for attr in ("kAXTitleAttribute", "kAXDescriptionAttribute", "kAXValueAttribute"):
                ref = self._attr(element, attr)
                if ref:
                    try:
                        text = self._string(ref) or text
                    finally:
                        self._ax.cf.CFRelease(ref)
                    if text:
                        break
            text = (text or "").strip()
            if not text:
                return

            pos = self._attr(element, "kAXPositionAttribute")
            size = self._attr(element, "kAXSizeAttribute")
            try:
                if not pos or not size:
                    return
                point = (ctypes.c_float * 2)()
                extent = (ctypes.c_float * 2)()
                # kAXValueCGPointType=1, kAXValueCGSizeType=2
                if not self._ax.app.AXValueGetValue(ctypes.c_void_p(pos), 1, point):
                    return
                if not self._ax.app.AXValueGetValue(ctypes.c_void_p(size), 2, extent):
                    return
                left, top = float(point[0]), float(point[1])
                width, height = float(extent[0]), float(extent[1])
            finally:
                if pos:
                    self._ax.cf.CFRelease(pos)
                if size:
                    self._ax.cf.CFRelease(size)
            if width <= 0 or height <= 0 or abs(left) > 30000 or abs(top) > 30000:
                return

            enabled_ref = self._attr(element, "kAXEnabledAttribute")
            is_enabled = None
            if enabled_ref:
                try:
                    is_enabled = bool(self._ax.cf.CFBooleanGetValue(ctypes.c_void_p(enabled_ref)))
                except Exception:
                    pass
                finally:
                    self._ax.cf.CFRelease(enabled_ref)

            bounds = (left, top, left + width, top + height)
            cx, cy = (bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0
            key = (text.lower(), int(cx), int(cy))
            if key not in seen:
                seen.add(key)
                out.append(Element(text=text, confidence=1.0, center=(cx, cy),
                                   bounds=bounds, control_type=role, is_enabled=is_enabled))
        finally:
            for child in self._array(element, "kAXChildrenAttribute"):
                try:
                    self._walk(child, depth + 1, out, seen)
                finally:
                    self._ax.cf.CFRelease(child)
