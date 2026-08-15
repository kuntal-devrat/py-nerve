"""Manual desktop-automation tester — deterministic, no AI.

Drives Py-Nerve's core directly (OCR + native input) so you can verify that
detection and clicking are correct *before* trusting the AI agent. If this
script clicks the right things, mistakes come from the LLM's planning; if it
also misclicks, there is a core bug (check the ``dpi`` command first).

Commands (read-only unless stated):
    find <text>                Locate an element: coords, bounds, confidence
    all <text>                 Print every match (ambiguity check)
    list [n]                   Print the first n on-screen elements as OCR sees them
    boxes [--text T] [out.png] Save annotated screenshot: what OCR sees
    dpi                        Report display scaling (logical vs physical px)
    move <text>                Move the cursor there (NO click)
    click <text>               Move + click (asks for confirmation)
    type_into <label> <text>   Click the field and type (confirmation)
    scroll <amount>            Scroll the wheel (positive = up)
    interactive                Prompt loop (default when no command given)

Use --dry-run to preview every action without touching the mouse.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Any

import pynerve as nv


def _print_element(el: Any, index: int | None = None) -> None:
    tag = f"[{index}] " if index is not None else ""
    text = el["text"] if isinstance(el, dict) else el.text
    conf = el["confidence"] if isinstance(el, dict) else el.confidence
    center = el["center"] if isinstance(el, dict) else el.center
    bounds = el["bounds"] if isinstance(el, dict) else el.bounds
    print(
        f"{tag}text={text!r} conf={conf} "
        f"center=({center[0]:.0f},{center[1]:.0f}) "
        f"bounds={tuple(int(v) for v in bounds)}"
    )


def _confirm(what: str, x: float, y: float, dry_run: bool, no_confirm: bool) -> bool:
    if dry_run:
        print(f"[dry-run] would {what} at ({x:.0f},{y:.0f})")
        return False
    if no_confirm:
        return True
    return input(f"{what} at ({x:.0f},{y:.0f})? [y/N] ").strip().lower() in ("y", "yes")


def cmd_find(text: str) -> None:
    el = nv.find(text)
    _print_element(el)
    print(f"screen: {nv.screenshot().size}")


def cmd_list(n: int) -> None:
    elements = nv.observe()
    print(f"{len(elements)} elements on screen; first {n}:")
    for i, el in enumerate(elements[:n]):
        _print_element(el, i)


def cmd_all(text: str) -> None:
    matches = nv.find_all(text)
    if not matches:
        print(f"no matches for {text!r}")
        return
    print(f"{len(matches)} match(es) for {text!r}:")
    for i, el in enumerate(matches):
        _print_element(el, i)


def cmd_boxes(out: str, highlight: str | None, dry_run: bool) -> None:
    from PIL import ImageDraw, ImageFont

    img = nv.screenshot()
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    elements = nv.observe()
    hit: dict[tuple[float, float], dict] = {}
    if highlight:
        for el in nv.find_all(highlight):
            hit[(round(el.center[0]), round(el.center[1]))] = el

    for el in elements:
        left, top, right, bottom = (int(v) for v in el["bounds"])
        color = "red" if (round(el["center"][0]), round(el["center"][1])) in hit else "lime"
        draw.rectangle([left, top, right, bottom], outline=color, width=2)
        cx, cy = int(el["center"][0]), int(el["center"][1])
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=color)
        label = el["text"][:40]
        draw.text((left, max(0, top - 12)), label, fill=color, font=font)
    img.save(out)
    print(f"saved {out} — {len(elements)} elements drawn "
          f"(red = {highlight!r} matches)" if highlight else f"saved {out} — {len(elements)} elements drawn")
    if dry_run:
        print("[dry-run] nothing else was done")


def cmd_dpi() -> None:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "$r = Get-ItemProperty 'HKCU:\\Control Panel\\Desktop' -Name LogPixels -ErrorAction SilentlyContinue; "
             "$d = Get-CimInstance Win32_VideoController | Select-Object -First 1 CurrentHorizontalResolution, CurrentVerticalResolution; "
             "Write-Host ('LogPixels=' + $r.LogPixels + ' Scale=' + [math]::Round(100 * $r.LogPixels / 96, 0) + '%'); "
             "Write-Host ('Physical=' + $d.CurrentHorizontalResolution + 'x' + $d.CurrentVerticalResolution)"],
            capture_output=True, text=True, timeout=30,
        )
        print(out.stdout.strip() or out.stderr.strip())
    except Exception as e:  # noqa: BLE001
        print(f"could not read DPI: {e}")
    print(f"PIL/logical screen: {nv.screenshot().size}")
    print("If Scale > 100%, OCR pixels can exceed logical mouse coordinates — a core mismatch.")


def cmd_move(text: str, dry_run: bool) -> None:
    el = nv.find(text)
    if dry_run:
        print(f"[dry-run] would move to {text!r} at ({el.center[0]:.0f},{el.center[1]:.0f})")
        return
    print(f"moving to {text!r} at ({el.center[0]:.0f},{el.center[1]:.0f}) — watch the cursor")
    nv.bezier_move(el.center[0], el.center[1], 0.6)


def cmd_click(text: str, dry_run: bool, no_confirm: bool) -> None:
    el = nv.find(text)
    if _confirm(f"click {text!r}", el.center[0], el.center[1], dry_run, no_confirm):
        nv.click(text)
        print("clicked")


def cmd_type_into(label: str, content: str, dry_run: bool, no_confirm: bool) -> None:
    el = nv.find(label)
    if _confirm(f"type {content!r} into {label!r}", el.center[0], el.center[1], dry_run, no_confirm):
        nv.type_into(label, content)
        print("typed")


def interactive(args: argparse.Namespace) -> None:
    print("interactive — type a command (find/all/move/click/type_into/scroll/boxes/dpi/quit)")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or line.lower() in ("quit", "exit", "q"):
            break
        parts = line.split(maxsplit=1)
        dispatch(parts[0].lower(), parts[1] if len(parts) > 1 else "", args)


def dispatch(cmd: str, rest: str, args: argparse.Namespace) -> None:
    try:
        if cmd == "find":
            cmd_find(rest)
        elif cmd == "all":
            cmd_all(rest)
        elif cmd == "list":
            cmd_list(int(rest) if rest.strip() else 15)
        elif cmd == "boxes":
            parts = rest.split()
            out = parts[0] if parts else args.out
            cmd_boxes(out, args.text, args.dry_run)
        elif cmd == "dpi":
            cmd_dpi()
        elif cmd == "move":
            cmd_move(rest, args.dry_run)
        elif cmd == "click":
            cmd_click(rest, args.dry_run, args.no_confirm)
        elif cmd == "type_into":
            label, _, content = rest.partition(" ")
            cmd_type_into(label, content, args.dry_run, args.no_confirm)
        elif cmd == "scroll":
            nv.scroll(int(rest)) if not args.dry_run else print(f"[dry-run] would scroll {rest}")
        else:
            print(f"unknown command: {cmd!r}")
    except nv.ElementNotFoundError as e:
        print(f"NOT FOUND: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {type(e).__name__}: {e}")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")

    ap = argparse.ArgumentParser(description="Manual desktop automation tester (no AI)")
    ap.add_argument("--dry-run", action="store_true", help="preview actions, never touch the mouse")
    ap.add_argument("--no-confirm", action="store_true", help="skip y/N confirmation for actions")
    ap.add_argument("--text", default=None, help="highlight this text in the boxes screenshot")
    ap.add_argument("--out", default="boxes.png", help="output file for the boxes screenshot")
    ap.add_argument("command", nargs="*", help="command + arguments, e.g. find File")
    args = ap.parse_args(argv)

    if not args.command:
        interactive(args)
        return 0
    cmd = args.command[0].lower()
    rest = " ".join(args.command[1:])
    dispatch(cmd, rest, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
