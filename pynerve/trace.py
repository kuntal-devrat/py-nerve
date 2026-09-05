"""JSONL action tracing with an HTML replay viewer.

Every action performed through a traced :class:`pynerve.PyNerve` instance
(``PyNerve(trace_path="run.jsonl")``) appends one JSON object per line::

    {"seq": 1, "action": "click", "args": {"text": "Save"},
     "started": "2026-09-05T07:00:00", "elapsed_ms": 412.3,
     "ok": true, "error": null}

:func:`render_html` converts a trace file into a standalone HTML report with
a summary table and per-action timing bars. Stdlib only.
"""

from __future__ import annotations

import functools
import html
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Callable


class ActionTracer:
    """Append-only JSONL tracer; safe to share across threads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._seq = 0
        if self.path.parent != Path(""):
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        action: str,
        args: dict[str, Any] | None = None,
        ok: bool = True,
        error: str | None = None,
        elapsed_ms: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one event and return it."""
        with self._lock:
            self._seq += 1
            event: dict[str, Any] = {
                "seq": self._seq,
                "action": action,
                "args": args or {},
                "started": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "elapsed_ms": round(elapsed_ms, 1),
                "ok": ok,
                "error": error,
            }
            if extra:
                event["extra"] = extra
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, default=str) + "\n")
            return event

    def wrap(self, fn: Callable[..., Any], name: str | None = None) -> Callable[..., Any]:
        """Wrap a callable so calls are traced (used by ``PyNerve``)."""
        action = name if name else str(getattr(fn, "__name__", "action"))

        @functools.wraps(fn)
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            start = monotonic()
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                self.log(action, _safe_args(args, kwargs), ok=False,
                         error=f"{type(e).__name__}: {e}",
                         elapsed_ms=(monotonic() - start) * 1000.0)
                raise
            self.log(action, _safe_args(args, kwargs), ok=True,
                     elapsed_ms=(monotonic() - start) * 1000.0)
            return result

        return _wrapped

    def read(self) -> list[dict[str, Any]]:
        """Read all events (skips blank/corrupt lines)."""
        if not self.path.exists():
            return []
        events = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return events


def _safe_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Represent positional args without leaking huge reprs."""
    safe: dict[str, Any] = {}
    for i, value in enumerate(args):
        text = repr(value)
        safe[f"arg{i}"] = text if len(text) <= 300 else text[:300] + "…"
    for key, value in kwargs.items():
        text = repr(value)
        safe[key] = text if len(text) <= 300 else text[:300] + "…"
    return safe


def render_html(events: list[dict[str, Any]], title: str = "Dexflow action trace") -> str:
    """Render events as a standalone HTML report."""
    total = len(events)
    failed = sum(1 for e in events if not e.get("ok", True))
    elapsed = sum(float(e.get("elapsed_ms") or 0) for e in events)
    max_ms = max([float(e.get("elapsed_ms") or 0) for e in events] + [1.0])

    rows = []
    for e in events:
        ms = float(e.get("elapsed_ms") or 0)
        width = max(2.0, 100.0 * ms / max_ms)
        status = "ok" if e.get("ok", True) else "fail"
        args = html.escape(json.dumps(e.get("args", {}), default=str)[:220])
        err = html.escape(str(e.get("error") or ""))
        rows.append(
            f"<tr class='{status}'><td>{e.get('seq', '')}</td>"
            f"<td>{html.escape(str(e.get('action', '')))}</td>"
            f"<td class='args'>{args}</td>"
            f"<td class='bar'><div style='width:{width:.1f}%'></div></td>"
            f"<td class='num'>{ms:.0f}</td>"
            f"<td>{status}</td><td class='args'>{err}</td></tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;color:#222}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ddd;padding:6px 8px;text-align:left;font-size:13px}}
th{{background:#f4f4f4}}tr.fail{{background:#fdecec}}
.args{{font-family:monospace;max-width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.num{{text-align:right}}.bar div{{background:#4a90d9;height:10px}}
tr.fail .bar div{{background:#d94a4a}}
.summary{{margin-bottom:1rem}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="summary"><strong>{total}</strong> actions, <strong>{failed}</strong> failed,
total <strong>{elapsed / 1000.0:.1f}s</strong>.</p>
<table><tr><th>#</th><th>action</th><th>args</th><th></th><th>ms</th><th>status</th><th>error</th></tr>
{''.join(rows) if rows else '<tr><td colspan="7">(no events)</td></tr>'}
</table></body></html>
"""


def render_html_file(trace_path: str | Path, out_path: str | Path | None = None) -> Path:
    """Render a ``.jsonl`` trace to an HTML file next to it (or ``out_path``)."""
    trace_path = Path(trace_path)
    events = ActionTracer(trace_path).read()
    out = Path(out_path) if out_path else trace_path.with_suffix(".html")
    out.write_text(render_html(events, title=f"Dexflow trace — {trace_path.name}"),
                   encoding="utf-8")
    return out
