"""AI agent layer for Py-Nerve.

Py-Nerve executes desktop actions *deterministically* (OCR + native input);
an LLM only decides *what* to do next. That combination gives you the
reliability of a script with the adaptability of an agent — no pixel
guessing, no paid "computer use" API.

The agent speaks to any OpenAI-compatible chat-completions endpoint:

- OpenAI:            https://api.openai.com/v1          (model="gpt-4o-mini")
- Ollama (local):    http://localhost:11434/v1          (model="llama3.2")
- LM Studio / Groq / OpenRouter / vLLM / ...  anything with /chat/completions

Example:
    from pynerve import run_agent

    result = run_agent(
        "Open the Settings app and enable dark mode",
        model="llama3.2",
        base_url="http://localhost:11434/v1",
        dry_run=True,   # plan only, don't touch the mouse
    )
    print(result.final_answer)

Zero new dependencies: the HTTP client is stdlib.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from .exceptions import PyNerveError

logger = logging.getLogger("pynerve.agent")

# Default OpenAI-compatible endpoint (OpenAI). Override with base_url= for
# local models: http://localhost:11434/v1 (Ollama), http://localhost:1234/v1 (LM Studio).
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"

# Sentinel returned by ``done`` / ``fail`` tools so the loop can detect them.
_DONE_SENTINEL = "__PYNERVE_DONE__"
_FAIL_SENTINEL = "__PYNERVE_FAIL__"


class AgentError(PyNerveError):
    """Raised when the agent loop fails (timeout, bad response, disallowed tool...)."""


@dataclass
class ToolSpec:
    """A callable action exposed to the LLM as a function-calling tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class AgentResult:
    """Outcome of an agent run."""

    task: str
    success: bool
    final_answer: str
    steps: int
    transcript: list[dict] = field(default_factory=list)


def _tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    fn: Callable[..., Any],
) -> ToolSpec:
    return ToolSpec(name, description, parameters, fn)


def _interruptible_sleep(duration: float) -> None:
    """Sleep in 100ms slices so Ctrl+C (SIGINT) is handled immediately on all platforms."""
    end = time.monotonic() + duration
    while time.monotonic() < end:
        time.sleep(min(0.1, max(0.0, end - time.monotonic())))


# ---------------------------------------------------------------------------
# Observe formatting helpers
# ---------------------------------------------------------------------------


def _filter_noise(elements: list[dict]) -> list[dict]:
    """Remove invalid or empty OCR artifacts while preserving UI buttons and numbers."""
    filtered: list[dict] = []
    for el in elements:
        text = el.get("text", "").strip()
        conf = el.get("confidence", 0)
        if not text:
            continue
        # Only drop extremely low confidence text artifacts
        if conf < 0.25:
            continue
        filtered.append(el)
    return filtered


def _dedup_nearby(elements: list[dict], px_threshold: int = 10) -> list[dict]:
    """Deduplicate elements with identical text within *px_threshold* pixels."""
    seen: dict[str, tuple[float, float]] = {}
    result: list[dict] = []
    for el in elements:
        key = el["text"].lower().strip()
        cx, cy = el["center"][0], el["center"][1]
        if key in seen:
            ox, oy = seen[key]
            if abs(cx - ox) < px_threshold and abs(cy - oy) < px_threshold:
                continue
        seen[key] = (cx, cy)
        result.append(el)
    return result


def _format_element_rows(items: list[dict]) -> list[str]:
    if not items:
        return []
    items_sorted = sorted(items, key=lambda e: (e["center"][1], e["center"][0]))
    rows: list[list[dict]] = []
    current_row: list[dict] = [items_sorted[0]]
    for el in items_sorted[1:]:
        if abs(el["center"][1] - current_row[-1]["center"][1]) < 25:
            current_row.append(el)
        else:
            rows.append(current_row)
            current_row = [el]
    rows.append(current_row)

    lines: list[str] = []
    for row in rows:
        y = int(row[0]["center"][1])
        parts: list[str] = []
        for el in row:
            text = el["text"]
            ct = el.get("control_type")
            val = el.get("value")
            prefix = f"[{ct}] " if ct else ""
            suffix = f" (val='{val}')" if val else ""
            parts.append(f"{prefix}'{text}'{suffix}")
        lines.append(f"  row y≈{y}: {' | '.join(parts)}")
    return lines


def _format_observe(elements: list[dict], max_elements: int = 150) -> str:
    """Render the observed layout as semantic rows enriched with UIA control types.

    Prioritizes elements in the active foreground window to prevent confusion
    with background windows or desktop noise.
    """
    items = _filter_noise(elements)
    items = _dedup_nearby(items)
    items = items[:max_elements]

    if not items:
        return "(no elements detected on screen)"

    active_items = [el for el in items if el.get("in_active_window") is True]
    bg_items = [el for el in items if el.get("in_active_window") is not True]

    lines: list[str] = []
    if active_items:
        lines.append(f"Screen elements in ACTIVE WINDOW ({len(active_items)} items):")
        lines.extend(_format_element_rows(active_items))
        if bg_items:
            lines.append(f"\nOther desktop / background elements ({len(bg_items)} items):")
            lines.extend(_format_element_rows(bg_items[:25]))
    else:
        lines.append(f"Screen elements ({len(items)} items):")
        lines.extend(_format_element_rows(items))

    if len(elements) > len(items):
        lines.append(f"  ({len(elements) - len(items)} noise elements filtered)")

    return "\n".join(lines)




def _compute_observe_diff(
    before: list[dict] | None,
    after: list[dict],
    max_items: int = 20,
) -> str:
    """Compute what changed between two observe snapshots."""
    if before is None:
        return ""

    before_set = {el["text"].lower().strip() for el in before if el.get("text")}
    after_set = {el["text"].lower().strip() for el in after if el.get("text")}

    added = sorted(after_set - before_set)[:max_items]
    removed = sorted(before_set - after_set)[:max_items]

    parts: list[str] = []
    if added:
        parts.append(f"NEW on screen: {added}")
    if removed:
        parts.append(f"GONE from screen: {removed}")
    if not parts:
        parts.append("(screen unchanged)")

    return "\nDIFF: " + " | ".join(parts)


# ---------------------------------------------------------------------------
# Dialog detection helper
# ---------------------------------------------------------------------------

_DIALOG_BUTTON_PATTERNS = {
    "ok", "cancel", "yes", "no", "close", "save", "don't save",
    "discard", "retry", "abort", "ignore", "apply", "continue",
    "accept", "decline", "allow", "deny", "open", "delete",
    "confirm", "dismiss",
}


def _detect_dialog(elements: list[dict]) -> dict | None:
    """Detect common dialog patterns in the current screen elements.

    Returns:
        Dict with 'buttons' and 'type' if a dialog is detected, else None.
    """
    button_texts: list[str] = []
    for el in elements:
        text_lower = el.get("text", "").lower().strip()
        if text_lower in _DIALOG_BUTTON_PATTERNS:
            button_texts.append(el["text"].strip())

    if len(button_texts) < 2:
        return None

    # Classify the dialog type
    button_set = {b.lower() for b in button_texts}
    if button_set & {"save", "don't save", "discard"}:
        dialog_type = "save_confirmation"
    elif button_set & {"yes", "no"}:
        dialog_type = "yes_no_confirmation"
    elif button_set & {"ok", "cancel"}:
        dialog_type = "ok_cancel"
    elif button_set & {"retry", "abort", "ignore"}:
        dialog_type = "error"
    elif button_set & {"allow", "deny"}:
        dialog_type = "permission"
    else:
        dialog_type = "unknown"

    return {"type": dialog_type, "buttons": button_texts}


# ---------------------------------------------------------------------------
# Build tools
# ---------------------------------------------------------------------------

def build_tools(nv: Any = None, max_observe_elements: int = 80, vision: bool = False) -> list[ToolSpec]:
    """Build the default tool set backed by a PyNerve instance.

    Args:
        nv: A PyNerve instance (or anything with the same methods).
            Defaults to the module-level singleton.
        max_observe_elements: Cap on how many on-screen elements are sent to
            the LLM per ``observe`` call (keeps local-model context small).
        vision: If True, include ``screenshot_base64`` for vision-capable LLMs.

    Returns:
        List of ToolSpec objects usable with ``Agent(tools=...)``.
    """
    from . import _get_nv

    nv = nv or _get_nv()

    # Mutable state for observe-diff tracking (shared across tool calls)
    _observe_state: dict[str, Any] = {"last": None}

    def _observe_with_diff() -> str:
        raw = nv.observe()
        formatted = _format_observe(raw, max_elements=max_observe_elements)
        diff = _compute_observe_diff(_observe_state["last"], raw)
        _observe_state["last"] = raw
        return formatted + diff

    tools = [
        _tool(
            "observe",
            "Return the current on-screen elements as grouped rows of text labels. "
            "Call this FIRST before any action, and AFTER every action to verify the result. "
            "Elements are grouped by visual rows (menu bars, toolbars, content, etc.).",
            {
                "type": "object",
                "properties": {},
            },
            lambda: _observe_with_diff(),
        ),
        _tool(
            "click",
            "Move the mouse to an on-screen element by its EXACT text label (from observe output) "
            "and left-click it. Use relative_to/direction when several elements share the same label.",
            {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "EXACT text label from observe output"},
                    "relative_to": {"type": "string", "description": "Optional anchor label for relative positioning"},
                    "direction": {"type": "string", "enum": ["right", "left", "above", "below"], "description": "Direction from the anchor"},
                },
                "required": ["text"],
            },
            lambda text, **kw: nv.click(text, **kw),
        ),
        _tool(
            "double_click",
            "Move to an element by its EXACT text label and double-click it.",
            {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "EXACT text label from observe output"}},
                "required": ["text"],
            },
            lambda text, **kw: nv.double_click(text, **kw),
        ),
        _tool(
            "right_click",
            "Move to an element by its EXACT text label and right-click it (opens context menu).",
            {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "EXACT text label from observe output"}},
                "required": ["text"],
            },
            lambda text, **kw: nv.right_click(text, **kw),
        ),
        _tool(
            "middle_click",
            "Move to an element by its EXACT text label and middle-click it.",
            {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "EXACT text label from observe output"}},
                "required": ["text"],
            },
            lambda text, **kw: nv.middle_click(text, **kw),
        ),
        _tool(
            "hover",
            "Move mouse cursor over an on-screen element and dwell there without clicking.",
            {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "EXACT text label from observe output"},
                    "dwell": {"type": "number", "description": "Seconds to dwell at target (default: 0.2)"},
                },
                "required": ["text"],
            },
            lambda text, **kw: nv.hover(text, **kw),
        ),
        _tool(
            "get_clipboard",
            "Read current text content from the system clipboard.",
            {
                "type": "object",
                "properties": {},
            },
            lambda: nv.get_clipboard(),
        ),
        _tool(
            "set_clipboard",
            "Write text to the system clipboard.",
            {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Text to set on clipboard"}},
                "required": ["text"],
            },
            lambda text: nv.set_clipboard(text),
        ),
        _tool(
            "type_into",
            "Click an input field identified by its EXACT text label and type content into it. "
            "Set clear=true to select-all and delete existing content first.",
            {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "EXACT label of the input field from observe output"},
                    "content": {"type": "string", "description": "Text to type into the field"},
                    "clear": {"type": "boolean", "description": "Select all + delete existing content first"},
                },
                "required": ["text", "content"],
            },
            lambda text, content, **kw: nv.type_into(text, content, **kw),
        ),
        _tool(
            "type_text",
            "Type raw text at the current cursor position (no clicking — use after clicking a field).",
            {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Text to type"}},
                "required": ["text"],
            },
            lambda text: nv.type_text(text),
        ),
        _tool(
            "press_key",
            "Press a single key: enter, tab, escape, backspace, space, up, down, left, right, "
            "home, end, delete, f1-f12, etc.",
            {
                "type": "object",
                "properties": {"key": {"type": "string", "description": "Key name (e.g. 'enter', 'tab', 'escape')"}},
                "required": ["key"],
            },
            lambda key: nv.press_key(key),
        ),
        _tool(
            "key_combo",
            "Press a key combination. Examples: ['ctrl', 's'] for save, ['alt', 'tab'] to switch windows, "
            "['ctrl', 'a'] to select all, ['ctrl', 'c'] to copy.",
            {
                "type": "object",
                "properties": {"keys": {"type": "array", "items": {"type": "string"}, "description": "Keys to press together"}},
                "required": ["keys"],
            },
            lambda keys: nv.key_combo(keys),
        ),
        _tool(
            "scroll",
            "Scroll the mouse wheel at the current position. Positive = scroll up, negative = scroll down. "
            "Use -3 or -5 for scrolling down through content.",
            {
                "type": "object",
                "properties": {"amount": {"type": "integer", "description": "Wheel notches (positive=up, negative=down)"}},
                "required": ["amount"],
            },
            lambda amount: nv.scroll(amount),
        ),
        _tool(
            "scroll_to",
            "Scroll until an element with the given text becomes visible on screen. "
            "Useful when the target is below the fold.",
            {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to scroll to"},
                    "timeout": {"type": "number", "description": "Max seconds to scroll (default 15)"},
                },
                "required": ["text"],
            },
            lambda text, **kw: nv.scroll_to(text, **kw),
        ),
        _tool(
            "wait_for",
            "Wait until an element with the given text appears on screen. "
            "Use after actions that trigger loading (page navigation, app launch, dialog open).",
            {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to wait for"},
                    "timeout": {"type": "number", "description": "Max seconds to wait (default 30)"},
                },
                "required": ["text"],
            },
            lambda text, **kw: nv.wait_for(text, **kw),
        ),
        _tool(
            "wait",
            "Pause execution for a given number of seconds (e.g. 1.0 or 2.0) to allow dynamic web pages, "
            "search results, or animations to finish rendering before observing.",
            {
                "type": "object",
                "properties": {
                    "seconds": {"type": "number", "description": "Seconds to wait (default: 1.0)"},
                },
            },
            lambda seconds=1.0: [time.sleep(min(max(float(seconds), 0.1), 30.0)), f"Waited {seconds}s for UI to settle."][1], # type: ignore
        ),
        _tool(
            "find",
            "Check whether an element with the given text exists on screen and report its position. "
            "Does NOT click — use click() to interact.",
            {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            lambda text: str(nv.find(text)),
        ),
        _tool(
            "focus_window",
            "Bring a window whose title contains the given text to the foreground. "
            "Use after launching an app or when switching between windows.",
            {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Text contained in the window title"},
                    "timeout": {"type": "number", "description": "Max seconds to wait (default 10)"},
                },
                "required": ["title"],
            },
            lambda title, **kw: nv.focus_window(title, **kw),
        ),
        _tool(
            "launch",
            "Launch an application, open a file, or open a URL using the OS app launcher. "
            "Examples: 'notepad', 'chrome', 'https://www.youtube.com', "
            "'C:\\\\Program Files\\\\...\\\\app.exe'. "
            "Use this FIRST when the target app is not already open, then call focus_window + observe.",
            {
                "type": "object",
                "properties": {
                    "app": {"type": "string", "description": "App name, executable path, document, or URL"}
                },
                "required": ["app"],
            },
            lambda app: nv.launch(app),
        ),
        # -- Coordinate tools (for non-text elements when using vision mode) --
        _tool(
            "click_at",
            "Click at specific pixel coordinates (x, y). Use ONLY when you identified a non-text "
            "element (icon, checkbox, image) via screenshot_base64 and know its coordinates. "
            "Prefer click() with text labels whenever possible.",
            {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X pixel coordinate"},
                    "y": {"type": "integer", "description": "Y pixel coordinate"},
                    "button": {"type": "string", "enum": ["left", "right"], "description": "Mouse button (default: left)"},
                },
                "required": ["x", "y"],
            },
            lambda x, y, button="left": _click_at_impl(nv, x, y, button),
        ),
        # -- Completion tools --
        _tool(
            "done",
            "Call this when the task is COMPLETE. Provide a summary of what was accomplished. "
            "This is the correct way to finish — do NOT just stop calling tools.",
            {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "What was accomplished"},
                },
                "required": ["summary"],
            },
            lambda summary: f"{_DONE_SENTINEL}{summary}",
        ),
        _tool(
            "fail",
            "Call this when the task is IMPOSSIBLE or cannot be completed. Explain why.",
            {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why the task cannot be completed"},
                },
                "required": ["reason"],
            },
            lambda reason: f"{_FAIL_SENTINEL}{reason}",
        ),
        # -- Intelligence tools --
        _tool(
            "detect_dialog",
            "Check if a dialog box (Save/OK/Cancel/Yes/No/etc.) is currently visible on screen. "
            "Returns the dialog type and available buttons, or null if no dialog detected.",
            {
                "type": "object",
                "properties": {},
            },
            lambda: _detect_dialog_impl(nv),
        ),
        _tool(
            "explore_by_scrolling",
            "Scroll through content to discover all elements (useful for long lists or pages). "
            "Returns a combined list of all unique elements seen across multiple scroll positions.",
            {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["down", "up"], "description": "Scroll direction (default: down)"},
                    "pages": {"type": "integer", "description": "Number of pages to scroll (default: 3)"},
                },
            },
            lambda direction="down", pages=3: _explore_scroll_impl(nv, direction, pages, max_observe_elements),
        ),
    ]

    # Vision tool: only include when using a vision-capable model
    if vision:
        tools.append(
            _tool(
                "screenshot_base64",
                "Capture a screenshot of the current screen and return it as a base64-encoded JPEG. "
                "Use this when observe() text alone is insufficient — e.g. to identify icons, images, "
                "colors, or layout that isn't captured by OCR text. The image will be shown in the "
                "conversation so you can describe what you see and then act on it.",
                {
                    "type": "object",
                    "properties": {},
                },
                lambda: _screenshot_b64_impl(nv),
            )
        )

    return tools


def _click_at_impl(nv: Any, x: int, y: int, button: str = "left") -> str:
    """Click at specific pixel coordinates."""
    from .input import bezier_move
    from .input import click as _click
    bezier_move(float(x), float(y), nv.move_duration)
    _click(button)
    nv.invalidate_cache()
    return f"Clicked at ({x}, {y}) with {button} button"


def _detect_dialog_impl(nv: Any) -> str:
    """Detect dialog boxes on the current screen."""
    elements = nv.observe()
    result = _detect_dialog(elements)
    if result is None:
        return "No dialog detected on screen."
    return json.dumps(result)


def _explore_scroll_impl(
    nv: Any, direction: str = "down", pages: int = 3, max_elements: int = 80,
) -> str:
    """Scroll through content and collect all unique elements."""
    import time as _time
    all_elements: dict[str, dict] = {}
    amount = -3 if direction == "down" else 3

    for _ in range(pages):
        elements = nv.observe()
        for el in elements:
            key = el["text"].lower().strip()
            if key and key not in all_elements:
                all_elements[key] = el
        nv.scroll(amount)
        _time.sleep(0.5)

    combined = list(all_elements.values())
    return _format_observe(combined, max_elements=max_elements)


def _screenshot_b64_impl(nv: Any) -> str:
    """Capture screen and return base64 JPEG."""
    img = nv.screenshot()
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=60)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"__SCREENSHOT_B64__{b64}"


# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Configuration for an agent run."""

    model: str = DEFAULT_MODEL
    base_url: str | None = None  # None -> $PYNERVE_BASE_URL or OpenAI
    api_key: str | None = None  # None -> $PYNERVE_API_KEY / $OPENAI_API_KEY / $GROQ_API_KEY / $GOOGLE_API_KEY
    system_prompt: str | None = None
    temperature: float = 0.2
    max_steps: int = 25
    timeout: float = 120.0
    dry_run: bool = False  # plan only: print actions, never execute
    allowlist: list[str] | None = None  # None = all tools allowed
    max_observe_elements: int = 80
    reasoning_effort: str | None = None  # e.g. "low" for gpt-oss on Groq (saves tokens)
    retry_refusals: bool = True  # nudge the model again when it "refuses" without acting
    vision: bool = False  # include screenshot_base64 for vision-capable LLMs
    max_history_messages: int = 14  # context window compression threshold
    step_delay: float = 0.0  # seconds to wait between LLM requests (e.g. 15.0 for Groq free-tier rate limits)
    max_tokens: int | None = 1024  # token reservation cap (prevents 402 errors on OpenRouter)


    def resolve(self) -> "AgentConfig":
        if self.base_url is None:
            self.base_url = os.environ.get("PYNERVE_BASE_URL", DEFAULT_BASE_URL)
        if self.api_key is None:
            self.api_key = (
                os.environ.get("PYNERVE_API_KEY")
                or os.environ.get("OPENROUTER_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("GROQ_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
            )
        return self


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """\
You are Py-Nerve, a desktop automation agent. You control a real computer by \
calling tools that interact with on-screen text elements detected by OCR and UI Automation.

## CRITICAL RULES — FOLLOW THESE EXACTLY

1. **ALWAYS call `observe` FIRST** before any action. You are BLIND without it.
2. **FOCUS ON ACTIVE WINDOW**: The observe output highlights elements in the \
ACTIVE WINDOW. Always target controls inside the active window.
3. **PREFER KEYBOARD FOR INPUT & CALCULATIONS**:
   - When entering numbers, equations, math formulas (e.g. 256 * 4), search terms, or \
text into an active application (Calculator, Notepad, terminal, search box), \
PREFER typing directly with `type_text` (e.g. `type_text(text="256*4=")` or `type_into`) \
instead of clicking individual number/letter buttons one by one. Keyboard typing is 10x faster and 100% accurate.
4. **USE `click` FOR CONTROLS & BUTTONS**:
   - Use `click` to activate UI buttons (like 'Submit', 'Open', 'Save', 'OK'), menu items, tabs, or checkboxes.
5. **ONLY use EXACT text** from the observe output as tool arguments. \
NEVER invent or guess text labels. Copy them exactly.
6. **After EVERY action**, call `observe` to verify the screen changed as expected before taking the next step.
7. **If the target app is not open**, call `launch` first, then `wait_for` or `focus_window`, then `observe`.
8. **When the task is complete**, call the `done` tool with a summary. \
**When the task is impossible**, call the `fail` tool with a reason.
9. **If a dialog box appears** unexpectedly (OK/Cancel/Save/etc.), handle \
it first — click the appropriate button or press Escape to dismiss it.
10. **CALL ONLY ONE (1) TOOL PER TURN (STRICT ReAct)**:
    - Call exactly ONE tool per response. Never return a long list of future tool calls.
    - After taking an action (launching, typing, clicking, pressing Enter), wait for the tool output in the next turn so you can see what actually happened.
11. **SPATIAL HIERARCHY (Header vs Main Content)**:
    - In web apps and desktop software (YouTube, Google, Spotify, File Explorer), the search box and navigation controls are in the top header (`y < 120`).
    - The actual search results, videos, tracks, and documents appear below in the main content area (`y > 150`).
    - When asked to open or play a result, select a result item from the content area (`y > 150`), NOT the search bar at `y ≈ 50`.
12. **READING & SUMMARIZING CONTENT**:
    - When asked to read a webpage (Wikipedia, news, docs) and write its summary into Notepad or another app:
      1. Open the page and call `observe` to read the text.
      2. Read and synthesize the summary directly from the `observe` text in your context.
      3. Focus the target app: `focus_window("Notepad")`.
      4. Paste the summary: `set_clipboard(text="<Your concise summary>")` → `key_combo(["ctrl", "v"])` (or `type_text`).
      5. DO NOT try to click webpage buttons like 'Edit' or 'Select all'.

## OBSERVE OUTPUT FORMAT

`observe` returns text elements grouped into visual rows, prioritizing the active window:
```
Screen elements in ACTIVE WINDOW (24 items):
  row y≈12: [Window] 'Calculator'
  row y≈160: [Edit] 'Display is 0' (val='0')
  row y≈340: [Button] '7' | [Button] '8' | [Button] '9' | [Button] 'Multiply'
  row y≈440: [Button] '1' | [Button] '2' | [Button] '3' | [Button] 'Plus'
  row y≈490: [Button] '0' | [Button] 'Decimal' | [Button] 'Equals'
```

## COMMON TASK PATTERNS

**Calculations & Math in Calculator:**
1. launch(app="Calculator") → observe
2. type_text(text="256*4=") → observe (read display result)
3. done(summary="Calculated 256 * 4 = 1024")

**Opening an app and editing text:**
1. launch(app="notepad") → wait_for(text="Untitled") → observe
2. type_text(text="Hello world") → observe
3. key_combo(keys=["ctrl", "s"]) → wait_for(text="Save As")

**Searching on YouTube or the Web:**
1. launch(app="https://www.youtube.com") → wait(seconds=2.0) → observe
2. type_into(text="Search", content="mala song") → press_key(key="enter")
3. wait(seconds=1.5) → observe (observe the loaded video results below y > 150)
4. click(text="<Exact Video Title from search results>") → done(summary="Playing video")

**Summarizing a web article into Notepad:**
1. launch(app="notepad")
2. launch(app="https://www.wikipedia.org")
3. type_into(text="Search", content="Artificial intelligence")
4. press_key(key="enter")
5. observe() -> Read the article text. Call scroll(amount=-5) and observe again to read more sections.
6. focus_window(title="Untitled")
7. set_clipboard(text="Artificial Intelligence (AI) is the intelligence of machines...")
8. key_combo(keys=["ctrl", "v"])
9. observe() -> Verify text is visible in Notepad.
10. key_combo(keys=["ctrl", "s"])
11. wait(seconds=1.0)
12. observe() -> Verify Save As dialog is open.
13. type_into(text="File name:", content="AI.txt")
14. click(text="Save")
15. observe() -> Verify window title updated to AI.txt - Notepad.
16. done(summary="Researched Wikipedia, composed summary, and saved to AI.txt")"""



# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """ReAct loop: observe → LLM proposes → Py-Nerve executes → repeat.

    Deterministic execution, LLM planning. No new dependencies; works with any
    OpenAI-compatible chat-completions endpoint (OpenAI, Ollama, LM Studio...).
    """

    def __init__(
        self,
        nv: Any | None = None,
        tools: list[ToolSpec] | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        from . import _get_nv

        self.nv = nv or _get_nv()
        self.config = (config or AgentConfig()).resolve()
        self.tools = tools if tools is not None else build_tools(
            self.nv, self.config.max_observe_elements, self.config.vision,
        )

        self._tool_map = {t.name: t for t in self.tools}
        if self.config.allowlist is not None:
            self._tool_map = {
                name: t for name, t in self._tool_map.items() if name in self.config.allowlist
            }

        # State for observe-diff and error tracking
        self._consecutive_errors: int = 0
        self._pending_screenshot_b64: str | None = None

    def run(self, task: str) -> AgentResult:
        """Execute ``task`` and return the result."""
        cfg = self.config
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": cfg.system_prompt or DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Task: {task}"},
        ]
        transcript: list[dict] = []

        for step in range(1, cfg.max_steps + 1):
            logger.info("[agent] step %d/%d: %s", step, cfg.max_steps, task)

            # Delay between requests (useful for Groq free-tier rate limits)
            if step > 1 and cfg.step_delay > 0:
                logger.info("[agent] waiting %.1fs before next request...", cfg.step_delay)
                _interruptible_sleep(cfg.step_delay)


            # Compress history if it's getting too long (prevents context overflow)
            messages = self._compress_history(messages, cfg.max_history_messages)

            response = self._call_llm(messages)

            message = response["choices"][0]["message"]
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                # Final answer, or JSON actions in content for models without native
                # function-calling (small local models often emit fenced blocks).
                actions = self._parse_content_actions(content)
                if actions:
                    parts = []
                    for a in actions:
                        result_str = self._execute_action(a, transcript, step)
                        # Check for done/fail sentinel
                        done_result = self._check_sentinel(result_str, task, step, transcript)
                        if done_result is not None:
                            return done_result
                        parts.append(result_str)
                    messages.extend(
                        [
                            {"role": "assistant", "content": content},
                            {"role": "tool", "tool_call_id": f"step{step}", "content": "\n".join(parts)[:8000]},
                        ]
                    )
                    # Inject reflection after consecutive errors
                    self._maybe_inject_recovery(messages, task, step, cfg.max_steps)
                    continue
                if not content.strip() or (cfg.retry_refusals and _is_refusal(content)):
                    if step < cfg.max_steps:
                        logger.info("[agent] empty or refusal response, asking the model to continue")
                        messages.extend(
                            [
                                {"role": "assistant", "content": content or "I observed the screen."},
                                {
                                    "role": "user",
                                    "content": (
                                        "That did not complete the task. Please proceed by calling an action tool "
                                        "(e.g. `type_into`, `click`, `press_key`, `wait`, `observe`), "
                                        "or call `done(summary=...)` if the task is finished."
                                    ),
                                },
                            ]
                        )
                        continue
                return AgentResult(
                    task=task, success=True, final_answer=content.strip(), steps=step, transcript=transcript
                )

            # Enforce single-turn ReAct execution: if the model emitted a batch of
            # speculative tool calls, execute only the first one so the agent observes
            # real screen feedback at every step instead of hallucinating future UI states.
            active_calls = tool_calls[:1]
            messages.append({"role": "assistant", "content": content, "tool_calls": active_calls})
            for call in active_calls:
                fn = call.get("function") or {}
                fn_name = fn.get("name")
                if not fn_name:
                    # Some providers (e.g. Cloudflare Workers AI) occasionally
                    # emit tool calls with an empty function object; surface it
                    # as a recoverable error instead of crashing the loop.
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", f"step{step}"),
                            "content": (
                                "ERROR: malformed tool call (missing function name). "
                                "Call a valid tool with the exact name from the tool list, "
                                "or call the `done` tool to finish."
                            ),
                        }
                    )
                    continue
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result_str = self._execute_action({"name": fn_name, "arguments": args}, transcript, step)

                # Check for done/fail sentinel
                done_result = self._check_sentinel(result_str, task, step, transcript)
                if done_result is not None:
                    return done_result

                # Handle screenshot: inject as vision content in next LLM call
                tool_content = result_str
                if result_str.startswith("__SCREENSHOT_B64__"):
                    self._pending_screenshot_b64 = result_str[len("__SCREENSHOT_B64__"):]
                    tool_content = "Screenshot captured. I can see the screen now."

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", f"step{step}"),
                        "content": tool_content[:4000],
                    }
                )

            # Inject reflection / recovery after consecutive errors
            self._maybe_inject_recovery(messages, task, step, cfg.max_steps)

        raise AgentError(
            f"Agent exceeded max_steps={cfg.max_steps} without finishing the task. "
            f"Transcript: {transcript}"
        )

    # -- internals -----------------------------------------------------------

    def _check_sentinel(self, result_str: str, task: str, step: int, transcript: list[dict]) -> AgentResult | None:
        """Check if a tool result contains a done/fail sentinel."""
        if result_str.startswith(_DONE_SENTINEL):
            summary = result_str[len(_DONE_SENTINEL):]
            return AgentResult(task=task, success=True, final_answer=summary, steps=step, transcript=transcript)
        if result_str.startswith(_FAIL_SENTINEL):
            reason = result_str[len(_FAIL_SENTINEL):]
            return AgentResult(task=task, success=False, final_answer=reason, steps=step, transcript=transcript)
        return None

    def _maybe_inject_recovery(self, messages: list[dict], task: str, step: int, max_steps: int) -> None:
        """Inject recovery guidance after consecutive errors."""
        if self._consecutive_errors >= 3 and step < max_steps:
            logger.warning("[agent] %d consecutive errors — injecting recovery guidance", self._consecutive_errors)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"⚠️ You have made {self._consecutive_errors} consecutive errors. "
                        "STOP and reassess. Steps:\n"
                        "1. Call `observe` to see the current screen state.\n"
                        "2. Check if a dialog or popup is blocking — if so, press Escape or "
                        "click Cancel/OK to dismiss it.\n"
                        "3. Try a COMPLETELY DIFFERENT approach to the task.\n"
                        f"4. If the task is impossible, call `fail` with a reason.\n"
                        f"\nOriginal task: {task}"
                    ),
                }
            )
            self._consecutive_errors = 0  # reset after injection

    def _execute_action(self, action: dict, transcript: list[dict], step: int) -> str:
        name = action["name"]
        args = action.get("arguments") or {}
        spec = self._tool_map.get(name)
        if spec is None:
            self._consecutive_errors += 1
            return f"ERROR: unknown or disallowed tool '{name}'. Available: {sorted(self._tool_map)}"

        transcript.append({"step": step, "tool": name, "args": args})
        logger.info("[agent] step %d: %s(%s)", step, name, json.dumps(args))
        if self.config.dry_run:
            self._consecutive_errors = 0
            return f"DRY-RUN: would call {name}({json.dumps(args)})"

        try:
            out = spec.fn(**args)
            if self.config.step_delay > 0:
                time.sleep(self.config.step_delay)
        except Exception as e:
            self._consecutive_errors += 1
            return f"ERROR: {type(e).__name__}: {e}"

        self._consecutive_errors = 0
        return str(out)[:4000]

    def _call_llm(self, messages: list[dict]) -> dict[str, Any]:
        cfg = self.config
        base_url = cfg.base_url or DEFAULT_BASE_URL
        url = base_url.rstrip("/") + "/chat/completions"

        # Prepare messages — inject pending screenshot as vision content
        send_messages = list(messages)
        if self._pending_screenshot_b64 and cfg.vision:
            b64 = self._pending_screenshot_b64
            self._pending_screenshot_b64 = None
            # Add the screenshot as a user message with image content
            send_messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here is the current screenshot. Describe what you see and decide what to do next."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                        },
                    ],
                }
            )

        body: dict[str, Any] = {
            "model": cfg.model,
            "messages": send_messages,
            "temperature": cfg.temperature,
        }
        tools = [t.to_openai_schema() for t in self._tool_map.values()]
        if tools:
            body["tools"] = tools
        if cfg.max_tokens is not None:
            body["max_tokens"] = cfg.max_tokens
        if cfg.reasoning_effort:
            body["reasoning_effort"] = cfg.reasoning_effort

        # Some providers (e.g. Groq behind Cloudflare) reject the default
        # Python-urllib User-Agent with HTTP 403 "error code: 1010"; a
        # browser-like UA passes their bot filter and is harmless elsewhere.
        headers = {
            "Content-Type": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            ),
            "HTTP-Referer": "https://github.com/kuntal-devrat/py-nerve",
            "X-Title": "Dexflow",
        }
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"

        # Retry transient failures: rate limits (429/5xx) and provider-side tool-call
        # parse failures (400 tool_use_failed), which happen occasionally with
        # reasoning models like gpt-oss and usually succeed on the next draw.
        last_error: AgentError | None = None
        for attempt in range(3):
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
                    data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
                    return data
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")
                if e.code == 401 and not cfg.api_key:
                    detail += (
                        " — no API key was provided. Pass api_key= (or --api-key) "
                        "or set PYNERVE_API_KEY / OPENAI_API_KEY / GROQ_API_KEY."
                    )
                # Provider-side tool-call parse failure (e.g. Groq 400 tool_use_failed)
                # Groq returns the raw generation in `failed_generation` when its server-side
                # parser fails on Llama-3 XML tags. We recover and execute the tool call directly.
                if e.code == 400 and "tool_use_failed" in detail:
                    try:
                        err_json = json.loads(detail)
                        failed_gen = err_json.get("error", {}).get("failed_generation")
                        if failed_gen:
                            recovered_actions = Agent._parse_content_actions(failed_gen)
                            if recovered_actions:
                                logger.info("[agent] Recovered tool call from provider failed_generation: %s", recovered_actions)
                                tool_calls = [
                                    {
                                        "id": f"call_groq_rec_{i}",
                                        "type": "function",
                                        "function": {
                                            "name": act["name"],
                                            "arguments": json.dumps(act.get("arguments", {})),
                                        },
                                    }
                                    for i, act in enumerate(recovered_actions)
                                ]
                                return {
                                    "choices": [
                                        {
                                            "message": {
                                                "role": "assistant",
                                                "content": "",
                                                "tool_calls": tool_calls,
                                            }
                                        }
                                    ]
                                }
                    except Exception as rec_err:
                        logger.debug("[agent] failed to recover from failed_generation: %s", rec_err)

                retryable = e.code in (429, 500, 502, 503, 504) or (
                    e.code == 400 and "tool_use_failed" in detail
                )
                last_error = AgentError(f"LLM endpoint {url} returned HTTP {e.code}: {detail[:500]}")

                if not retryable or attempt == 2:
                    raise last_error from e
                logger.warning(
                    "[agent] LLM call HTTP %d (attempt %d/3), retrying: %s",
                    e.code, attempt + 1, detail[:200],
                )
                time.sleep(0.5 * (attempt + 1))

            except urllib.error.URLError as e:
                raise AgentError(f"Could not reach LLM endpoint {url}: {e}") from e
        if last_error is not None:
            raise last_error
        raise AgentError(f"Failed to communicate with LLM endpoint {url}")  # pragma: no cover

    @staticmethod
    def _compress_history(messages: list[dict], max_messages: int) -> list[dict]:
        """Compress conversation history to prevent context window overflow.

        Keeps system prompt + original task + the most recent messages.
        Old observe/tool results are summarized to save tokens.
        """
        if len(messages) <= max_messages:
            return messages

        # Always keep: system prompt (index 0) + original task (index 1)
        head = messages[:2]

        # Keep the most recent messages in full
        tail_count = max_messages - 2
        tail = messages[-tail_count:]

        # Summarize the middle section
        middle = messages[2:-tail_count]
        if middle:
            # Count how many tool calls were in the middle
            tool_count = sum(1 for m in middle if m.get("role") == "tool")
            action_count = sum(1 for m in middle if m.get("role") == "assistant" and m.get("tool_calls"))
            head.append(
                {
                    "role": "user",
                    "content": (
                        f"[Earlier in this session: {action_count} actions were taken and "
                        f"{tool_count} tool results were received. The details have been "
                        f"summarized to save context. Focus on the recent messages below.]"
                    ),
                }
            )

        return head + tail

    @staticmethod
    def _parse_content_actions(content: str) -> list[dict]:
        """Parse every JSON tool call embedded in plain content (for models
        without native function-calling, e.g. qwen2.5-coder:1.5b).

        Handles one or more fenced ```json blocks, a bare JSON object, or a
        JSON object embedded in prose. Returns [] when nothing parses.
        """
        text = (content or "").strip()
        actions: list[dict] = []
        seen: set[str] = set()

        def _add(parsed: dict | None) -> None:
            if parsed is None:
                return
            key = json.dumps(parsed, sort_keys=True)
            if key not in seen:
                seen.add(key)
                actions.append(parsed)

        for block in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL):
            _add(Agent._try_parse_action(block.group(1).strip()))

        # 2) Groq / Llama 3 function tags:
        #    <function=launch{"app": "calculator"}</function>
        #    <function=launch>{"app": "calculator"}</function>
        for m in re.finditer(r"<function=([a-z_][a-z0-9_]*)(?:>\s*(.*?)\s*</function>|\s*(\{.*?\})\s*</function>)", text, flags=re.IGNORECASE | re.DOTALL):
            tag = m.group(1).lower()
            inner = (m.group(2) or m.group(3) or "").strip()
            args: dict | None = None
            obj = _first_json_object(inner)
            if obj is not None:
                try:
                    parsed = json.loads(obj)
                    if isinstance(parsed, dict):
                        args = parsed
                except json.JSONDecodeError:
                    pass
            if args is None:
                args = {"text": inner} if inner else {}
            _add({"name": tag, "arguments": args})

        # 3) Tool call tags (Hermes / Qwen / Mistral / DeepSeek):
        #    <tool_call>{"name": "click", "arguments": {"text": "OK"}}</tool_call>
        for m in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.IGNORECASE | re.DOTALL):
            _add(Agent._try_parse_action(m.group(1).strip()))

        # 4) XML-style tags: <launch>{"app": "chrome"};</launch>
        for m in re.finditer(r"<([a-z_][a-z0-9_]*)>\s*(.*?)\s*</\1>", text, flags=re.IGNORECASE | re.DOTALL):
            tag = m.group(1).lower()
            if tag in ("tool_call", "function"):
                continue
            inner = m.group(2).strip()
            args = None
            obj = _first_json_object(inner)
            if obj is not None:
                try:
                    parsed = json.loads(obj)
                    if isinstance(parsed, dict):
                        args = parsed
                except json.JSONDecodeError:
                    pass
            if args is None:
                args = {"text": inner}
            _add({"name": tag, "arguments": args})

        if not actions:
            obj = _first_json_object(text)
            _add(Agent._try_parse_action(obj) if obj is not None else None)

        return actions

    @staticmethod
    def _parse_content_action(content: str) -> dict | None:
        """First JSON action embedded in ``content`` (see ``_parse_content_actions``)."""
        actions = Agent._parse_content_actions(content)
        return actions[0] if actions else None

    @staticmethod
    def _try_parse_action(s: str) -> dict | None:
        """Parse a single JSON tool-call object into (name, arguments)."""
        # Tolerate trailing junk after the object (e.g. "{"a":1};") by scanning
        # for the balanced JSON object first.
        s = s.strip()
        obj = _first_json_object(s)
        if obj is not None and obj != s:
            s = obj
        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        name = data.get("name") or data.get("tool")
        if not name or not isinstance(name, str):
            return None
        args = data.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        if not isinstance(args, dict):
            # Flat form: {"tool": "click", "text": "File"} -> args are the rest.
            args = {k: v for k, v in data.items() if k not in ("name", "tool")}
        return {"name": name, "arguments": args}


def _is_refusal(content: str) -> bool:
    """Heuristic: is this a short "I can't do that" style answer rather than a
    genuine completion? Used to retry flaky provider draws (e.g. Cloudflare)."""
    text = (content or "").strip().lower()
    if len(text) > 300:
        return False
    markers = (
        "incomplete", "insufficient", "more details", "more information",
        "clarify", "cannot", "can't", "unable to", "not enough", "lacking",
        "please provide", "please specify", "i'm not sure what", "i am not sure what",
        "as an ai", "i can't help", "cannot help",
    )
    return any(m in text for m in markers)


def _first_json_object(text: str) -> str | None:
    """Return the first balanced {...} region in ``text`` (handles nesting and
    braces inside strings), or None if there is no complete object."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def agent(task: str, **kwargs: Any) -> AgentResult:
    """Run an AI agent task against the desktop. One-shot convenience wrapper.

    Args:
        task: Natural-language description of what to do.
        **kwargs: Any AgentConfig field (model, base_url, api_key, dry_run,
                  max_steps, allowlist, system_prompt...).

    Returns:
        AgentResult with final_answer and transcript.

    Example:
        result = pynerve.agent(
            "Open a browser, go to youtube.com and search for 'lofi beats'",
            model="llama3.2",
            base_url="http://localhost:11434/v1",   # Ollama, fully local
            dry_run=True,
        )
    """
    known = {
        "model",
        "base_url",
        "api_key",
        "system_prompt",
        "temperature",
        "max_steps",
        "timeout",
        "dry_run",
        "allowlist",
        "max_observe_elements",
        "reasoning_effort",
        "retry_refusals",
        "vision",
        "max_history_messages",
        "step_delay",
    }
    nv = kwargs.pop("nv", None)
    tools = kwargs.pop("tools", None)
    extra = set(kwargs) - known
    if extra:
        raise TypeError(f"Unknown agent option(s): {sorted(extra)}")
    config = AgentConfig(**kwargs)
    return Agent(nv=nv, tools=tools, config=config).run(task)
