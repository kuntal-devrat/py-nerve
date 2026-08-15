#!/usr/bin/env python3
"""Self-hosted coding agent backed by Ollama (or any OpenAI-compatible LLM).

Give it a task; it plans, writes/edits files, runs commands (tests, typechecks,
builds), reads the results, and iterates until the task is done — all inside
the workspace. No API keys, no cloud: it talks to your local Ollama server.

    # install a model first (one time):
    ollama pull qwen2.5-coder:7b

    # run the agent:
    python scripts/coding_agent.py "Fix the failing test in tests/test_matcher.py"
    python scripts/coding_agent.py "Add a CLI flag --fast to the benchmark" --dry-run

The agent can use these tools:
    list_directory(path)  read_file(path)  write_file(path, content)
    edit_file(path, old_string, new_string)  search(pattern, path)
    run_command(command)  finish(summary)

Safety:
- All file access is sandboxed to the workspace (path escapes are refused).
- Commands run in the workspace with a timeout and a blocklist of destructive
  operations (sudo, git push, rm -rf /, ...). Pass --allow-destructive to lift
  the blocklist.
- --dry-run prints planned actions without touching the filesystem.

Depends only on the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("coding_agent")

DEFAULT_BASE_URL = "http://localhost:11434/v1"  # Ollama's OpenAI-compatible API
DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_SYSTEM_PROMPT = """You are a coding agent working in a real repository on the user's machine.
You plan, write code, and EXECUTE commands to verify your work. Rules:

1. Inspect before you edit: use list_directory/read_file/search to understand the codebase.
2. Make minimal, targeted changes. Match the project's existing conventions and dependencies.
3. After changing code, run the relevant verification: `pytest tests/ -q`, a typecheck, or a build.
4. Read command output carefully. If something fails, fix it and re-run — iterate until green.
5. Never invent tool results. If a command errors, report the actual error and adapt.
6. When the task is complete, call finish() with a concise summary of what you changed and how you verified it.
7. Do not run destructive commands (the runner blocks them anyway): no sudo, no git push, no rm -rf of anything outside the workspace."""


# ---------------------------------------------------------------- tooling

class ToolSpec:
    def __init__(self, name: str, description: str, parameters: dict, fn: Callable[..., Any]) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.fn = fn

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class WorkspaceError(Exception):
    """Path escape or blocked command."""


class CodingAgent:
    def __init__(
        self,
        workspace: Path,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        api_key: str | None = None,
        system_prompt: str | None = None,
        max_steps: int = 15,
        timeout: float = 180.0,
        command_timeout: float = 120.0,
        dry_run: bool = False,
        allow_destructive: bool = False,
    ) -> None:
        self.workspace = workspace.resolve()
        self.model = model
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY") or ""
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_steps = max_steps
        self.timeout = timeout
        self.command_timeout = command_timeout
        self.dry_run = dry_run
        self.allow_destructive = allow_destructive
        self.tools = self._build_tools()
        self._tool_map = {t.name: t for t in self.tools}
        self.transcript: list[dict] = []
        # Limit tool results sent back to the model (context window).
        self.max_result_chars = 8000

    # -- path sandbox --------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        p = (self.workspace / path).resolve()
        if p != self.workspace and self.workspace not in p.parents:
            raise WorkspaceError(f"path escapes the workspace: {path}")
        return p

    # -- tools ---------------------------------------------------------------

    def _build_tools(self) -> list[ToolSpec]:
        def list_directory(path: str = ".") -> str:
            p = self._resolve(path)
            if not p.is_dir():
                return f"ERROR: not a directory: {path}"
            entries = []
            for child in sorted(p.iterdir()):
                marker = "/" if child.is_dir() else ""
                try:
                    size = child.stat().st_size if child.is_file() else 0
                    entries.append(f"{child.name}{marker}  ({size} bytes)" if child.is_file() else f"{child.name}/")
                except OSError:
                    continue
            return "\n".join(entries[:200]) or "(empty)"

        def read_file(path: str) -> str:
            p = self._resolve(path)
            if not p.is_file():
                return f"ERROR: not a file: {path}"
            data = p.read_text(encoding="utf-8", errors="replace")
            if len(data) > 30000:
                data = data[:30000] + f"\n... (truncated, file is {p.stat().st_size} bytes)"
            return data

        def write_file(path: str, content: str) -> str:
            if self.dry_run:
                return f"DRY-RUN: would write {path}"
            p = self._resolve(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"wrote {path} ({len(content)} bytes)"

        def edit_file(path: str, old_string: str, new_string: str) -> str:
            if self.dry_run:
                return f"DRY-RUN: would edit {path}"
            p = self._resolve(path)
            if not p.is_file():
                return f"ERROR: not a file: {path}"
            text = p.read_text(encoding="utf-8")
            if old_string not in text:
                return f"ERROR: old_string not found in {path} (exact match required)"
            count = text.count(old_string)
            if count > 1 and old_string == new_string:
                return f"ERROR: old_string appears {count} times; make it unique"
            p.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
            return f"edited {path} (replaced 1 occurrence)"

        def search(pattern: str, path: str = ".") -> str:
            p = self._resolve(path)
            if not p.is_dir():
                return f"ERROR: not a directory: {path}"
            try:
                rx = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                return f"ERROR: bad regex: {e}"
            hits: list[str] = []
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "target", ".venv", "node_modules", "3rd_party")]
                for name in files:
                    if name.endswith((".pyc", ".pyo")):
                        continue
                    fp = Path(root) / name
                    try:
                        for i, line in enumerate(fp.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                            if rx.search(line):
                                rel = fp.relative_to(self.workspace)
                                hits.append(f"{rel}:{i}: {line.strip()[:160]}")
                                if len(hits) >= 100:
                                    return "\n".join(hits)
                    except OSError:
                        continue
            return "\n".join(hits) or "(no matches)"

        def run_command(command: str) -> str:
            blocked = self._blocked(command)
            if blocked:
                return f"ERROR: blocked command (contains: {blocked}). Remove it or rerun with --allow-destructive."
            if self.dry_run:
                return f"DRY-RUN: would run: {command}"
            try:
                proc = subprocess.run(
                    command,
                    shell=True,
                    cwd=self.workspace,
                    capture_output=True,
                    text=True,
                    timeout=self.command_timeout,
                )
            except subprocess.TimeoutExpired:
                return f"ERROR: command timed out after {self.command_timeout}s: {command}"
            except Exception as e:
                return f"ERROR: failed to run command: {e}"
            out = (proc.stdout or "") + ("\n[stderr]\n" + proc.stderr if proc.stderr else "")
            if len(out) > self.max_result_chars:
                out = out[: self.max_result_chars] + "\n... (truncated)"
            status = "OK" if proc.returncode == 0 else f"EXIT {proc.returncode}"
            return f"[{status}] $ {command}\n{out}"

        def finish(summary: str) -> str:
            return f"__FINISH__ {summary}"

        return [
            ToolSpec("list_directory", "List files/directories in a path (relative to workspace root).",
                     {"type": "object", "properties": {"path": {"type": "string", "default": "."}}}, list_directory),
            ToolSpec("read_file", "Read a file's contents (relative to workspace root).",
                     {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, read_file),
            ToolSpec("write_file", "Create or overwrite a file with the given content (relative to workspace root).",
                     {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}, write_file),
            ToolSpec("edit_file", "Replace an exact old_string with new_string in a file (one occurrence).",
                     {"type": "object", "properties": {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}, "required": ["path", "old_string", "new_string"]}, edit_file),
            ToolSpec("search", "Regex-search file contents under a path; returns 'file:line: text' matches.",
                     {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string", "default": "."}}, "required": ["pattern"]}, search),
            ToolSpec("run_command", "Run a shell command in the workspace root and return its output (timeout-limited).",
                     {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}, run_command),
            ToolSpec("finish", "Call when the task is complete; pass a summary of changes and verification.",
                     {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}, finish),
        ]

    # -- command safety ------------------------------------------------------

    def _blocked(self, command: str) -> str | None:
        if self.allow_destructive:
            return None
        lowered = command.lower()
        patterns = [
            "sudo", "doas", "pkexec",
            "git push", "git reset --hard", "git clean",
            "rm -rf /", "rm -fr /", "rm -rf c:", "rm -rf ~",
            "del /s", "format ", "mkfs", "dd if=",
            "shutdown", "reboot",
            ":(){",  # fork bomb
        ]
        for pat in patterns:
            if pat in lowered:
                return pat
        return None

    # -- LLM round trip ------------------------------------------------------

    def _call_llm(self, messages: list[dict]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
        }
        tools = [t.to_openai_schema() for t in self.tools]
        if tools:
            body["tools"] = tools

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
                return data
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"LLM endpoint returned HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not reach LLM at {self.base_url} ({e}). Is Ollama running? "
                "Start it with `ollama serve` and pull a model first."
            ) from e

    @staticmethod
    def _parse_content_actions(content: str) -> list[dict]:
        """Parse tool calls embedded in plain content.

        Smaller local models (e.g. qwen2.5-coder:1.5b) often skip native
        function-calling and instead emit one or more ```json fenced blocks
        with {"name": ..., "arguments": {...}}. Accept bare JSON objects too.
        """
        actions: list[dict] = []
        fences = re.findall(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        candidates = fences if fences else [content]
        for block in candidates:
            try:
                data = json.loads(block.strip())
            except json.JSONDecodeError:
                continue
            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    continue
                name = item.get("name") or item.get("tool")
                if isinstance(name, str):
                    args = item.get("arguments")
                    actions.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
        return actions

    # -- execution -----------------------------------------------------------

    def _execute_action(self, action: dict, step: int) -> str:
        name = action["name"]
        args = action.get("arguments") or {}
        spec = self._tool_map.get(name)
        if spec is None:
            return f"ERROR: unknown tool '{name}'. Available: {sorted(self._tool_map)}"
        self.transcript.append({"step": step, "tool": name, "args": args})
        logger.info("[agent] step %d: %s(%s)", step, name, json.dumps(args)[:200])
        try:
            out = spec.fn(**args)
        except WorkspaceError as e:
            out = f"ERROR: {e}"
        except Exception as e:
            out = f"ERROR: {type(e).__name__}: {e}"
        return str(out)[: self.max_result_chars]

    def run(self, task: str) -> dict:
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Task: {task}"},
        ]
        final = {"task": task, "done": False, "summary": "", "steps": 0, "transcript": self.transcript}
        # Repetition guard: small local models (e.g. qwen2.5-coder:1.5b)
        # degenerate into repeating the same tool call. After the third repeat
        # of an identical action, feed back a corrective error instead of
        # silently executing it again.
        action_counts: dict[str, int] = {}

        def _action_key(name: str, args: dict) -> str:
            return f"{name}({json.dumps(args, sort_keys=True, default=str)})"

        def _execute(name: str, args: dict, step: int) -> str:
            key = _action_key(name, args)
            action_counts[key] = action_counts.get(key, 0) + 1
            if action_counts[key] >= 3:
                return (
                    f"ERROR: you have already performed this exact action {action_counts[key]} times. "
                    "Do NOT repeat it. Take a different action, or call finish() if the task is done."
                )
            return self._execute_action({"name": name, "arguments": args}, step)

        for step in range(1, self.max_steps + 1):
            logger.info("[agent] step %d/%d", step, self.max_steps)
            response = self._call_llm(messages)
            message = response["choices"][0]["message"]
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                actions = self._parse_content_actions(content)
                if not actions:
                    # Model finished (or gave up) without calling finish().
                    final["done"] = True
                    final["summary"] = content.strip()
                    return final
                messages.append({"role": "assistant", "content": content})
                for action in actions:
                    result = _execute(action["name"], action.get("arguments") or {}, step)
                    if result.startswith("__FINISH__"):
                        final["done"] = True
                        final["summary"] = result[len("__FINISH__"):].strip()
                        return final
                    messages.append({"role": "tool", "tool_call_id": f"step{step}", "content": result})
                continue

            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            for call in tool_calls:
                fn_name = call["function"]["name"]
                try:
                    args = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = _execute(fn_name, args, step)
                if fn_name == "finish" and result.startswith("__FINISH__"):
                    final["done"] = True
                    final["summary"] = result[len("__FINISH__"):].strip()
                    return final
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", f"step{step}"),
                    "content": result[: self.max_result_chars],
                })

        final["summary"] = (
            f"Stopped: reached max_steps={self.max_steps} without finish(). "
            f"Steps taken: {[t['tool'] for t in self.transcript]}"
        )
        return final


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", help="Natural-language task to complete in the workspace")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default {DEFAULT_MODEL})")
    ap.add_argument("--base-url", default=None, help=f"OpenAI-compatible endpoint (default {DEFAULT_BASE_URL})")
    ap.add_argument("--workspace", default=None, help="Workspace root (default: repo root)")
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--command-timeout", type=float, default=120.0)
    ap.add_argument("--dry-run", action="store_true", help="Plan only; never touch the filesystem")
    ap.add_argument("--allow-destructive", action="store_true", help="Lift the command blocklist")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    workspace = Path(args.workspace).resolve() if args.workspace else Path(__file__).resolve().parent.parent
    agent = CodingAgent(
        workspace,
        model=args.model,
        base_url=args.base_url,
        max_steps=args.max_steps,
        command_timeout=args.command_timeout,
        dry_run=args.dry_run,
        allow_destructive=args.allow_destructive,
    )
    print(f"Workspace: {workspace}")
    print(f"Model:     {args.model}  (dry_run={args.dry_run})")
    print("=" * 70)
    result = agent.run(args.task)
    print("=" * 70)
    print("DONE" if result["done"] else "STOPPED")
    print(result["summary"])


if __name__ == "__main__":
    main()
