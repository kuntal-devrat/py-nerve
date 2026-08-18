"""Py-Nerve: Deterministic desktop automation framework for Python.

Uses high-speed local computer vision (OCR) to control desktop applications
using human-readable text labels instead of fragile pixel coordinates.

Example:
    import pynerve as nv

    nv.click("File")
    nv.click("Open")
    nv.type_into("File name:", "document.txt")
    nv.click("Open", relative_to="File name:", direction="right")
"""

import sys

if sys.platform == "win32":
    try:
        import ctypes
        # Try Per-Monitor V2 DPI awareness (-4)
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        try:
            # Fallback to Per-Monitor V1 (2)
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                # Fallback to System DPI Aware
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

from . import _native
from ._types import Element
from .agent import (
    Agent,
    AgentConfig,
    AgentError,
    AgentResult,
    ToolSpec,
    build_tools,
)
from .capture import ScreenCapture
from .core import PyNerve
from .exceptions import (
    CaptureError,
    ElementNotFoundError,
    InputError,
    PyNerveError,
    VisionError,
)
from .input import (
    bezier_move,
    drag_to,
    get_position,
    key_combo,
    mouse_down,
    mouse_up,
    move_to,
    press_key,
    type_text,
)
from .matcher import filter_by_direction, find_all_matches, find_match
from .vision import VisionEngine

__version__ = "0.1.1"
__all__ = [
    # Core
    "PyNerve",
    "configure",
    # Public action functions
    "click",
    "double_click",
    "right_click",
    "middle_click",
    "hover",
    "type_into",
    "find",
    "find_all",
    "screenshot",
    "capture_window",
    "get_position",
    "get_clipboard",
    "set_clipboard",
    "list_monitors",
    "wait_for",
    "focus_window",
    "scroll",
    "scroll_to",
    "launch",
    "drag_and_drop",
    "observe",
    "observe_window",
    "invalidate_cache",
    # Low-level input
    "move_to",
    "bezier_move",
    "press_key",
    "key_combo",
    "type_text",
    "mouse_down",
    "mouse_up",
    "drag_to",
    # Types
    "Element",
    # Exceptions
    "PyNerveError",
    "ElementNotFoundError",
    "VisionError",
    "CaptureError",
    "InputError",
    # Engines
    "VisionEngine",
    "ScreenCapture",
    # Native Rust core (advanced users; see _native.pyi)
    "_native",
    # Matcher
    "find_match",
    "find_all_matches",
    "filter_by_direction",
    # Agent
    "run_agent",
    "Agent",
    "AgentConfig",
    "AgentResult",
    "AgentError",
    "ToolSpec",
    "build_tools",
]



# Singleton instance for module-level convenience functions
_nv: PyNerve | None = None


def _get_nv() -> PyNerve:
    """Get or create the singleton PyNerve instance."""
    global _nv
    if _nv is None:
        _nv = PyNerve()
    return _nv


def configure(**kwargs) -> None:
    """Configure the global PyNerve instance.

    Args:
        **kwargs: Arguments passed to PyNerve.__init__().
                  Replaces the existing instance.
    """
    global _nv
    _nv = PyNerve(**kwargs)


def click(text: str | Element, **kwargs) -> bool:
    """Move to element and left-click.

    Args:
        text: The text label to click.
        **kwargs: Passed to PyNerve.click().

    Returns:
        True if successful.
    """
    return _get_nv().click(text, **kwargs)


def double_click(text: str | Element, **kwargs) -> bool:
    """Move to element and double-click.

    Args:
        text: The text label to double-click.
        **kwargs: Passed to PyNerve.double_click().

    Returns:
        True if successful.
    """
    return _get_nv().double_click(text, **kwargs)


def right_click(text: str | Element, **kwargs) -> bool:
    """Move to element and right-click.

    Args:
        text: The text label to right-click.
        **kwargs: Passed to PyNerve.right_click().

    Returns:
        True if successful.
    """
    return _get_nv().right_click(text, **kwargs)


def middle_click(text: str | Element, **kwargs) -> bool:
    """Move to element and middle-click.

    Args:
        text: The text label to middle-click.
        **kwargs: Passed to PyNerve.middle_click().

    Returns:
        True if successful.
    """
    return _get_nv().middle_click(text, **kwargs)


def hover(text: str | Element, dwell: float = 0.2, **kwargs) -> bool:
    """Move cursor to element and dwell there without clicking.

    Args:
        text: The text label to hover over.
        dwell: Seconds to pause/dwell at target location.
        **kwargs: Passed to PyNerve.hover().

    Returns:
        True if successful.
    """
    return _get_nv().hover(text, dwell=dwell, **kwargs)


def type_into(text: str | Element, content: str, **kwargs) -> bool:
    """Find an input field and type text into it.

    Args:
        text: The text label of the input field.
        content: The text to type.
        **kwargs: Passed to PyNerve.type_into().

    Returns:
        True if successful.
    """
    return _get_nv().type_into(text, content, **kwargs)


def find(text: str | Element, **kwargs) -> Element:
    """Locate an element and return its position info.

    Args:
        text: The text label to find.
        **kwargs: Passed to PyNerve.find().

    Returns:
        Element with text, confidence, center, and bounds.
    """
    return _get_nv().find(text, **kwargs)


def find_all(text: str, threshold: int | None = None) -> list[Element]:
    """Find all elements matching the target text.

    Args:
        text: The text to search for.
        threshold: Override default confidence threshold.

    Returns:
        List of matching Elements.
    """
    return _get_nv().find_all(text, threshold)


def screenshot(region=None):
    """Capture a screenshot.

    Args:
        region: Optional (x, y, width, height) tuple.

    Returns:
        PIL Image.
    """
    return _get_nv().screenshot(region)


def wait_for(text: str | Element, timeout: float = 30.0) -> Element:
    """Wait dynamically for an element containing text to load.

    Args:
        text: The text to search for.
        timeout: Maximum wait time in seconds.

    Returns:
        The matched Element.
    """
    return _get_nv().wait_for(text, timeout)


def focus_window(title_substring: str, class_name: str | None = None, timeout: float = 10.0) -> bool:
    """Find and bring an application window to the foreground.

    Args:
        title_substring: Text contained in the window title.
        class_name: Optional class name filter.
        timeout: Maximum time to wait for the window in seconds.

    Returns:
        True if the window was successfully focused.
    """
    return _get_nv().focus_window(title_substring, class_name, timeout)


def scroll(amount: int, axis: str = "vertical") -> None:
    """Scroll the mouse wheel. Positive for scrolling up, negative for scrolling down.

    Args:
        amount: Number of wheel notches.
        axis: "vertical" (default) or "horizontal".
    """
    _get_nv().scroll(amount, axis=axis)


def get_clipboard() -> str:
    """Read text from the system clipboard."""
    return _get_nv().get_clipboard()


def set_clipboard(text: str) -> None:
    """Write text to the system clipboard."""
    _get_nv().set_clipboard(text)


def list_monitors() -> list[tuple[int, str, bool, tuple[int, int, int, int]]]:
    """List all connected monitors with their (index, name, is_primary, (x, y, width, height))."""
    return _get_nv().list_monitors()


def capture_window(title_substring: str):
    """Capture a screenshot of a specific window by title substring."""
    return _get_nv().capture_window(title_substring)


def observe_window(title_substring: str) -> list[dict]:
    """Observe screen elements constrained to a specific window."""
    return _get_nv().observe_window(title_substring)


def scroll_to(text: str | Element, **kwargs) -> bool:
    """Scroll the mouse wheel until an element containing the target text is visible on screen.

    Args:
        text: Target text to find.
        **kwargs: Passed to PyNerve.scroll_to().
    """
    return _get_nv().scroll_to(text, **kwargs)


def launch(app: str) -> str:
    """Launch an application, open a file, or open a URL using the OS app launcher.

    Args:
        app: App name (e.g. "chrome"), executable path, document, or URL.

    Returns:
        A short confirmation string.
    """
    return _get_nv().launch(app)


def drag_and_drop(source_text: str, target_text: str, **kwargs) -> bool:
    """Drag an element and drop it onto another element.

    Args:
        source_text: Label text of the element to drag.
        target_text: Label text of the target location to drop it on.
        **kwargs: Passed to PyNerve.drag_and_drop().
    """
    return _get_nv().drag_and_drop(source_text, target_text, **kwargs)


def observe(region: tuple[int, int, int, int] | None = None) -> list[dict]:
    """Return a deduplicated snapshot of the current UI layout as plain dicts.

    This is the primary API for AI agents: it gives an LLM the full "screen
    state" (text + confidence + center + bounds) without requiring a query.

    Args:
        region: Optional (x, y, width, height) region to observe.

    Returns:
        List of dicts: {"text", "confidence", "center": [x, y], "bounds": [l, t, r, b]}.
    """
    return _get_nv().observe(region)


def invalidate_cache() -> None:
    """Clear cached screenshots and OCR layouts."""
    _get_nv().invalidate_cache()



def run_agent(task: str, **kwargs):
    """Run an AI agent task against the desktop. One-shot convenience wrapper.

    Uses an LLM to plan steps (``observe`` -> act -> verify) while Py-Nerve
    executes them deterministically. Works with any OpenAI-compatible endpoint:
    OpenAI, or fully local via Ollama/LM Studio (``base_url=...``).

    Args:
        task: Natural-language description of what to do.
        **kwargs: AgentConfig fields — model, base_url, api_key, dry_run,
                  max_steps, allowlist, system_prompt, ...

    Returns:
        AgentResult with final_answer and transcript.
    """
    from .agent import agent as _run_agent

    return _run_agent(task, **kwargs)

