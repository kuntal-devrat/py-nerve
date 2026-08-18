"""Dexflow: High-speed, deterministic desktop automation framework & AI agent execution layer for Python.

Usage:
    import dexflow as df

    df.click("Save")
    df.type_into("Search", "Hello World")
"""

from pynerve import (
    Agent,
    AgentConfig,
    AgentError,
    AgentResult,
    Element,
    PyNerve,
    ScreenCapture,
    ToolSpec,
    VisionEngine,
    bezier_move,
    build_tools,
    capture_window,
    click,
    double_click,
    drag_and_drop,
    drag_to,
    find,
    find_all,
    find_all_matches,
    find_match,
    focus_window,
    get_clipboard,
    hover,
    invalidate_cache,
    key_combo,
    launch,
    list_monitors,
    middle_click,
    mouse_down,
    mouse_up,
    move_to,
    observe,
    observe_window,
    press_key,
    right_click,
    run_agent,
    scroll,
    scroll_to,
    set_clipboard,
    type_into,
    type_text,
    wait_for,
)
from pynerve.exceptions import (
    CaptureError,
    ElementNotFoundError,
    InputError,
    PyNerveError,
    VisionError,
)

__version__ = "0.1.1"

# Alias PyNerve class to Dexflow for brand consistency
Dexflow = PyNerve
DexflowError = PyNerveError

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentError",
    "AgentResult",
    "Element",
    "PyNerve",
    "ScreenCapture",
    "ToolSpec",
    "VisionEngine",
    "bezier_move",
    "build_tools",
    "capture_window",
    "click",
    "double_click",
    "drag_and_drop",
    "drag_to",
    "find",
    "find_all",
    "find_all_matches",
    "find_match",
    "focus_window",
    "get_clipboard",
    "hover",
    "invalidate_cache",
    "key_combo",
    "launch",
    "list_monitors",
    "middle_click",
    "mouse_down",
    "mouse_up",
    "move_to",
    "observe",
    "observe_window",
    "press_key",
    "right_click",
    "run_agent",
    "scroll",
    "scroll_to",
    "set_clipboard",
    "type_into",
    "type_text",
    "wait_for",
    "CaptureError",
    "ElementNotFoundError",
    "InputError",
    "PyNerveError",
    "VisionError"
]
