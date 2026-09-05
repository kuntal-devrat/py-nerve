from __future__ import annotations

import time

from . import _native
from .exceptions import InputError


def get_position() -> tuple[float, float]:
    """Get current mouse cursor position.

    Returns:
        Tuple of (x, y) coordinates.
    """
    try:
        return _native.get_mouse_position()
    except Exception as e:
        raise InputError(f"Failed to get mouse position: {e}") from e


def move_to(x: float, y: float) -> None:
    """Instantly move cursor to absolute coordinates.

    Args:
        x: Target X coordinate.
        y: Target Y coordinate.
    """
    try:
        _native.move_to(x, y)
    except Exception as e:
        raise InputError(f"Failed to move mouse: {e}") from e


def bezier_move(x: float, y: float, duration: float = 0.4) -> bool:
    """Move cursor along a cubic bezier curve with human-like motion.

    Args:
        x: Target X coordinate.
        y: Target Y coordinate.
        duration: Movement duration in seconds (0.2 - 0.5 recommended).

    Returns:
        True if human interference was detected and recovered from, False otherwise.
    """
    try:
        return bool(_native.bezier_move(x, y, duration))
    except Exception as e:
        raise InputError(f"Failed to bezier move: {e}") from e


def click(button: str = "left") -> None:
    """Perform a mouse click.

    Args:
        button: "left", "right", or "middle". Default is "left".
    """
    try:
        _native.click(button)
    except Exception as e:
        raise InputError(f"Failed to click: {e}") from e


def double_click() -> None:
    """Perform a double click at current position."""
    try:
        _native.double_click()
    except Exception as e:
        raise InputError(f"Failed to double click: {e}") from e


def type_text(
    text: str, interval: float = 0.05, use_clipboard: bool | None = None
) -> None:
    """Type text at the current cursor position.

    Short strings are typed character-by-character (preserves clipboard).
    Long strings (>80 chars by default, or ``use_clipboard=True``) are pasted
    via the clipboard (``Ctrl+V`` / ``Cmd+V``) which is ~50x faster, with the
    original clipboard content restored afterwards.

    Args:
        text: The text to type.
        interval: Delay between characters in seconds. Default is 0.05.
        use_clipboard: Force clipboard paste (True), force char typing (False),
            or auto-select based on length (None, default).
    """
    import sys as _sys

    should_paste = use_clipboard if use_clipboard is not None else len(text) > 80
    if should_paste and text:
        try:
            previous = None
            try:
                previous = _native.get_clipboard()
            except Exception:
                previous = None
            try:
                _native.set_clipboard(text)
            except Exception as e:
                raise InputError(f"Failed to set clipboard for paste: {e}") from e
            try:
                import time as _time

                _time.sleep(0.05)
                paste_keys = ["meta", "v"] if _sys.platform == "darwin" else ["ctrl", "v"]
                _native.key_combo(paste_keys)
                _time.sleep(0.05)
            finally:
                # Restore the user's original clipboard (best effort).
                if previous is not None:
                    try:
                        _native.set_clipboard(previous)
                    except Exception:
                        pass
            return
        except InputError:
            raise
        except Exception as e:
            raise InputError(f"Failed to paste text: {e}") from e
    try:
        _native.type_text(text, interval)
    except Exception as e:
        raise InputError(f"Failed to type text: {e}") from e


def press_key(key: str) -> None:
    """Press a single key.

    Args:
        key: Key name (e.g., "enter", "tab", "escape", "backspace", "space",
             "up", "down", "left", "right", "home", "end", "f1"-"f12", etc.)
    """
    try:
        _native.press_key(key)
    except Exception as e:
        raise InputError(f"Failed to press key: {e}") from e


def key_combo(keys: list[str]) -> None:
    """Press a key combination (e.g., Ctrl+S, Alt+Tab).

    Args:
        keys: List of key names to press together.
              Example: ["ctrl", "s"] for Ctrl+S.
    """
    try:
        _native.key_combo(keys)
    except Exception as e:
        raise InputError(f"Failed to press key combo: {e}") from e


def scroll(amount: int, axis: str = "vertical") -> None:
    """Scroll the mouse wheel.

    Args:
        amount: Number of wheel notches. Positive scrolls up, negative down.
        axis: "vertical" (default) or "horizontal".
    """
    try:
        _native.scroll(amount, axis)
    except Exception as e:
        raise InputError(f"Failed to scroll mouse: {e}") from e


def mouse_down(button: str = "left") -> None:
    """Press down a mouse button without releasing it.

    Args:
        button: "left", "right", or "middle".
    """
    try:
        _native.mouse_down(button)
    except Exception as e:
        raise InputError(f"Failed to press mouse button: {e}") from e


def mouse_up(button: str = "left") -> None:
    """Release a mouse button.

    Args:
        button: "left", "right", or "middle".
    """
    try:
        _native.mouse_up(button)
    except Exception as e:
        raise InputError(f"Failed to release mouse button: {e}") from e


def drag_to(x: float, y: float, duration: float = 0.4) -> None:
    """Drag the mouse from its current position to (x, y) coordinates."""
    try:
        mouse_down("left")
        time.sleep(0.1)
        bezier_move(x, y, duration)
        time.sleep(0.1)
        mouse_up("left")
    except Exception as e:
        raise InputError(f"Failed to drag mouse: {e}") from e


def hover(x: float, y: float, dwell: float = 0.2, duration: float = 0.4) -> None:
    """Move cursor to (x, y) coordinates and dwell there.

    Args:
        x: Target X coordinate.
        y: Target Y coordinate.
        dwell: Seconds to pause/dwell at target location.
        duration: Movement duration in seconds.
    """
    try:
        bezier_move(x, y, duration)
        if dwell > 0:
            time.sleep(dwell)
    except Exception as e:
        raise InputError(f"Failed to hover: {e}") from e


def middle_click() -> None:
    """Perform a middle-button mouse click at current position."""
    click("middle")


def get_clipboard() -> str:
    """Read text from system clipboard.

    Returns:
        String content from clipboard.
    """
    try:
        return _native.get_clipboard()
    except Exception as e:
        raise InputError(f"Failed to get clipboard: {e}") from e


def set_clipboard(text: str) -> None:
    """Write text to system clipboard.

    Args:
        text: Text string to copy to clipboard.
    """
    try:
        _native.set_clipboard(text)
    except Exception as e:
        raise InputError(f"Failed to set clipboard: {e}") from e


def list_monitors() -> list[tuple[int, str, bool, tuple[int, int, int, int]]]:
    """List all connected monitors with their (index, name, is_primary, (x, y, width, height))."""
    try:
        return _native.list_monitors()
    except Exception as e:
        raise InputError(f"Failed to list monitors: {e}") from e

