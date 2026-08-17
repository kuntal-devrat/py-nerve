"""Tests for the self-hosted coding agent (mocked LLM — no Ollama needed)."""

from __future__ import annotations

import json

import pytest

from scripts.coding_agent import CodingAgent, WorkspaceError


def _tool_call(name: str, arguments: dict) -> dict:
    return {
        "id": f"call_{name}",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _message(content: str | None, tool_calls: list | None = None) -> dict:
    return {"message": {"content": content, "tool_calls": tool_calls or []}}


def _make_agent(workspace, responses: list[dict], **kwargs) -> tuple[CodingAgent, list[list[dict]]]:
    kwargs.setdefault("max_steps", 10)
    agent = CodingAgent(workspace, **kwargs)
    seen: list[list[dict]] = []

    def llm(messages: list[dict]) -> dict:
        seen.append(list(messages))
        return responses.pop(0)

    object.__setattr__(agent, "_call_llm", llm)
    return agent, seen


class TestAgentLoop:
    def test_writes_file_runs_command_and_finishes(self, tmp_path):
        responses = [
            {"choices": [_message("", [_tool_call("write_file", {"path": "greeting.txt", "content": "Hello agent"})])]},
            {"choices": [_message("", [_tool_call("run_command", {"command": "python -c \"print('ran')\""})])]},
            {"choices": [_message("", [_tool_call("finish", {"summary": "wrote greeting.txt and verified"})])]},
        ]
        agent, seen = _make_agent(tmp_path, responses)

        result = agent.run("create a greeting file")
        assert result["done"] is True
        assert "greeting.txt" in result["summary"]
        assert (tmp_path / "greeting.txt").read_text() == "Hello agent"
        # The LLM saw the write + command results before the finish call.
        tool_msgs = [m for m in seen[2] if m.get("role") == "tool"]
        assert len(tool_msgs) == 2

    def test_content_json_fallback(self, tmp_path):
        # Model without native tool calling: emits JSON in plain content.
        responses = [
            {"choices": [_message('{"name": "write_file", "arguments": {"path": "a.txt", "content": "x"}}')]},
            {"choices": [_message("done", [])]},
        ]
        agent, _ = _make_agent(tmp_path, responses)
        result = agent.run("write a.txt")
        assert (tmp_path / "a.txt").read_text() == "x"
        assert result["done"] is True

    def test_multiple_fenced_json_actions_executed_in_order(self, tmp_path):
        # Small local models emit several ```json fenced tool calls in one reply.
        content = (
            '```json\n{"name": "write_file", "arguments": {"path": "a.txt", "content": "first"}}\n```\n'
            'then:\n'
            '```json\n{"name": "write_file", "arguments": {"path": "b.txt", "content": "second"}}\n```'
        )
        responses = [
            {"choices": [_message(content)]},
            {"choices": [_message("done", [])]},
        ]
        agent, _ = _make_agent(tmp_path, responses)
        result = agent.run("write two files")
        assert (tmp_path / "a.txt").read_text() == "first"
        assert (tmp_path / "b.txt").read_text() == "second"
        assert result["done"] is True

    def test_max_steps_stops(self, tmp_path):
        responses = [
            {"choices": [_message("", [_tool_call("run_command", {"command": "echo x"})])]} for _ in range(20)
        ]
        agent, _ = _make_agent(tmp_path, responses, max_steps=3)
        result = agent.run("loop forever")
        assert result["done"] is False
        assert "max_steps" in result["summary"]

    def test_unknown_tool_returns_error(self, tmp_path):
        responses = [
            {"choices": [_message("", [_tool_call("hack", {})])]},
            {"choices": [_message("ok", [])]},
        ]
        agent, seen = _make_agent(tmp_path, responses)
        result = agent.run("whatever")
        assert result["done"] is True
        tool_msgs = [m for m in seen[1] if m.get("role") == "tool"]
        assert tool_msgs and "unknown tool" in tool_msgs[0]["content"]

    def test_repetition_guard_blocks_identical_actions(self, tmp_path):
        # A small model that repeats the same write_file over and over.
        responses = [
            {"choices": [_message("", [_tool_call("write_file", {"path": "x.txt", "content": "a"})])]}
            for _ in range(5)
        ] + [{"choices": [_message("done", [])]}]
        agent, seen = _make_agent(tmp_path, responses, max_steps=6)
        agent.run("write x.txt")
        assert (tmp_path / "x.txt").read_text() == "a"  # first write still happened
        # The 3rd identical call must have been blocked with the corrective error.
        all_tool_msgs = [m["content"] for batch in seen for m in batch if m.get("role") == "tool"]
        assert any("already performed this exact action" in c for c in all_tool_msgs)
        # and the corrective error is the last tool result seen by the model
        assert all_tool_msgs[-1].startswith("ERROR:")


class TestSandbox:
    def test_path_escape_refused(self, tmp_path):
        agent = CodingAgent(tmp_path)
        with pytest.raises(WorkspaceError):
            agent._resolve("../outside.txt")

    def test_absolute_path_outside_refused(self, tmp_path):
        agent = CodingAgent(tmp_path)
        with pytest.raises(WorkspaceError):
            agent._resolve(str(tmp_path.parent / "elsewhere"))

    def test_inside_workspace_ok(self, tmp_path):
        agent = CodingAgent(tmp_path)
        assert agent._resolve("sub/file.py") == (tmp_path / "sub/file.py").resolve()


class TestCommandBlocklist:
    def test_sudo_blocked(self, tmp_path):
        agent = CodingAgent(tmp_path)
        assert agent._blocked("sudo rm -rf /") is not None

    def test_git_push_blocked(self, tmp_path):
        agent = CodingAgent(tmp_path)
        assert agent._blocked("git push origin main") is not None

    def test_blocked_command_is_error_result(self, tmp_path):
        agent = CodingAgent(tmp_path)
        out = agent._tool_map["run_command"].fn("sudo reboot")
        assert "blocked" in out

    def test_allow_destructive_lifts_blocklist(self, tmp_path):
        agent = CodingAgent(tmp_path, allow_destructive=True)
        assert agent._blocked("git push") is None


class TestDryRun:
    def test_write_file_does_not_touch_disk(self, tmp_path):
        agent = CodingAgent(tmp_path, dry_run=True)
        out = agent._tool_map["write_file"].fn("nope.txt", "content")
        assert out.startswith("DRY-RUN")
        assert not (tmp_path / "nope.txt").exists()

    def test_run_command_does_not_execute(self, tmp_path):
        agent = CodingAgent(tmp_path, dry_run=True)
        out = agent._tool_map["run_command"].fn("echo hi > dry.txt")
        assert "DRY-RUN" in out
        assert not (tmp_path / "dry.txt").exists()
