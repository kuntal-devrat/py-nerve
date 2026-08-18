"""Tests for the AI agent layer (no network, no desktop needed)."""

from __future__ import annotations

import json

import pytest

from pynerve import Agent, AgentConfig, AgentError, build_tools


class FakeNv:
    """Minimal stand-in for PyNerve that records calls instead of touching the desktop."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def click(self, text, **kwargs):
        self.calls.append(("click", text, kwargs))
        return True

    def double_click(self, text, **kwargs):
        self.calls.append(("double_click", text, kwargs))
        return True

    def right_click(self, text, **kwargs):
        self.calls.append(("right_click", text, kwargs))
        return True

    def type_into(self, text, content, **kwargs):
        self.calls.append(("type_into", text, content, kwargs))
        return True

    def type_text(self, text):
        self.calls.append(("type_text", text))
        return True

    def press_key(self, key):
        self.calls.append(("press_key", key))
        return True

    def key_combo(self, keys):
        self.calls.append(("key_combo", keys))
        return True

    def scroll(self, amount):
        self.calls.append(("scroll", amount))
        return True

    def scroll_to(self, text, **kwargs):
        self.calls.append(("scroll_to", text, kwargs))
        return True

    def wait_for(self, text, **kwargs):
        self.calls.append(("wait_for", text, kwargs))
        return True

    def find(self, text, **kwargs):
        self.calls.append(("find", text, kwargs))
        return f"Element(text={text!r})"

    def observe(self):
        return [{"text": "File", "confidence": 0.99, "center": [10.0, 10.0], "bounds": [0, 0, 20, 20]}]

    def focus_window(self, title, **kwargs):
        self.calls.append(("focus_window", title, kwargs))
        return True

    def launch(self, app):
        self.calls.append(("launch", app))
        return f"Launched: {app}"


def _tool_call(name: str, arguments: dict) -> dict:
    return {
        "id": "call_1",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _message(content: str | None, tool_calls: list | None = None) -> dict:
    return {"message": {"content": content, "tool_calls": tool_calls or []}}


def _make_agent(fake: FakeNv, responses: list[dict], **cfg) -> tuple[Agent, list[list[dict]]]:
    """Build an Agent with a scripted LLM; records every messages payload it saw."""
    cfg = dict(cfg)
    cfg.setdefault("max_steps", 8)
    agent = Agent(nv=fake, config=AgentConfig(**cfg))
    seen: list[list[dict]] = []

    def llm(messages: list[dict]) -> dict:
        seen.append(list(messages))
        return responses.pop(0)

    object.__setattr__(agent, "_call_llm", llm)
    return agent, seen


class TestConfigResolution:
    def test_resolve_prefers_explicit_key_over_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        cfg = AgentConfig(api_key="sk-explicit").resolve()
        assert cfg.api_key == "sk-explicit"

    def test_resolve_accepts_groq_key_env(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-env")
        assert AgentConfig().resolve().api_key == "gsk-env"

    def test_resolve_accepts_google_key_env(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "aq-env")
        assert AgentConfig().resolve().api_key == "aq-env"

    def test_resolve_key_priority(self, monkeypatch):
        monkeypatch.setenv("PYNERVE_API_KEY", "sk-pynerve")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("GROQ_API_KEY", "gsk-groq")
        assert AgentConfig().resolve().api_key == "sk-pynerve"


class TestToolSchemas:
    def test_build_tools_has_core_actions(self):
        tools = build_tools(FakeNv())
        names = {t.name for t in tools}
        assert {"click", "type_into", "observe", "wait_for", "scroll_to", "find", "press_key", "launch"} <= names

    def test_schemas_are_valid_json_schema(self):
        for tool in build_tools(FakeNv()):
            schema = tool.to_openai_schema()
            assert schema["type"] == "function"
            fn = schema["function"]
            assert fn["name"]
            assert fn["description"]
            assert fn["parameters"]["type"] == "object"
            assert "properties" in fn["parameters"]


class TestAgentLoop:
    def test_executes_tool_then_final_answer(self):
        fake = FakeNv()
        responses = [
            {"choices": [_message("", [_tool_call("click", {"text": "File"})])]},
            {"choices": [_message("Done!", [])]},
        ]
        agent, _ = _make_agent(fake, responses)

        result = agent.run("click File")
        assert result.success
        assert result.final_answer == "Done!"
        assert fake.calls == [("click", "File", {})]
        assert result.transcript[0]["tool"] == "click"

    def test_dry_run_never_executes(self):
        fake = FakeNv()
        responses = [
            {"choices": [_message("", [_tool_call("click", {"text": "File"})])]},
            {"choices": [_message("All planned", [])]},
        ]
        agent, seen = _make_agent(fake, responses, dry_run=True)

        result = agent.run("click File")
        assert fake.calls == []  # nothing executed
        assert result.success
        # The LLM's next view includes the DRY-RUN tool result.
        tool_msgs = [m for m in seen[1] if m.get("role") == "tool"]
        assert tool_msgs and "DRY-RUN" in tool_msgs[0]["content"]

    def test_malformed_tool_call_recovers(self):
        # Cloudflare Workers AI sometimes returns tool_calls with an empty
        # function object; the agent must surface an error and keep going.
        fake = FakeNv()
        responses = [
            {"choices": [_message("", [{"id": "c1", "type": "function", "function": {}}])]},
            {"choices": [_message("", [_tool_call("click", {"text": "File"})])]},
            {"choices": [_message("done", [])]},
        ]
        agent, seen = _make_agent(fake, responses)

        result = agent.run("do it")
        assert result.success
        assert fake.calls == [("click", "File", {})]  # recovered on the next draw
        tool_msgs = [m for m in seen[1] if m.get("role") == "tool"]
        assert tool_msgs and "malformed" in tool_msgs[0]["content"]

    def test_unknown_tool_result_is_error_message(self):
        fake = FakeNv()
        responses = [
            {"choices": [_message("", [_tool_call("hack", {})])]},
            {"choices": [_message("ok", [])]},
        ]
        agent, seen = _make_agent(fake, responses)

        result = agent.run("whatever")
        assert result.success
        tool_msgs = [m for m in seen[1] if m.get("role") == "tool"]
        assert tool_msgs and "ERROR" in tool_msgs[0]["content"]
        assert fake.calls == []

    def test_allowlist_filters_tools(self):
        fake = FakeNv()
        responses = [
            {"choices": [_message("", [_tool_call("click", {"text": "File"})])]},
            {"choices": [_message("", [_tool_call("type_text", {"text": "x"})])]},
            {"choices": [_message("done", [])]},
        ]
        agent, _ = _make_agent(fake, responses, allowlist=["click"])
        result = agent.run("task")
        assert result.success
        assert any(c[0] == "click" for c in fake.calls)
        assert not any(c[0] == "type_text" for c in fake.calls)

    def test_max_steps_raises(self):
        fake = FakeNv()
        responses = [
            {"choices": [_message("", [_tool_call("click", {"text": "File"})])]} for _ in range(20)
        ]
        agent, _ = _make_agent(fake, responses, max_steps=3)
        with pytest.raises(AgentError):
            agent.run("never finishes")

    def test_launch_tool_executes(self):
        fake = FakeNv()
        responses = [
            {"choices": [_message("", [_tool_call("launch", {"app": "chrome"})])]},
            {"choices": [_message("Chrome is open", [])]},
        ]
        agent, _ = _make_agent(fake, responses)
        result = agent.run("open chrome")
        assert result.success
        assert fake.calls == [("launch", "chrome")]

    def test_plain_text_final_answer(self):
        agent = Agent(nv=FakeNv(), config=AgentConfig(max_steps=3))
        agent._call_llm = lambda messages: {"choices": [_message("Task impossible", [])]}
        result = agent.run("do something impossible")
        assert result.success
        assert result.final_answer == "Task impossible"
        assert result.steps == 1

    def test_refusal_is_retried_not_final(self):
        # Cloudflare's model sometimes answers with a refusal-like text instead
        # of calling a tool; the agent must nudge and continue, not stop.
        fake = FakeNv()
        responses = [
            {"choices": [_message("Your input is incomplete. Please provide more details.", [])]},
            {"choices": [_message("", [_tool_call("observe", {})])]},
            {"choices": [_message("Screen checked", [])]},
        ]
        agent, seen = _make_agent(fake, responses)

        result = agent.run("check the screen")
        assert result.success
        assert result.final_answer == "Screen checked"
        assert any(t["tool"] == "observe" for t in result.transcript)
        # The nudge was sent after the refusal.
        assert any(m["role"] == "user" and "did not complete" in m["content"] for m in seen[1])

    def test_refusal_retry_can_be_disabled(self):
        fake = FakeNv()
        responses = [{"choices": [_message("Your input is insufficient.", [])]}]
        agent, _ = _make_agent(fake, responses, retry_refusals=False)
        result = agent.run("whatever")
        assert result.success
        assert result.final_answer == "Your input is insufficient."
        assert result.steps == 1


class TestRetry:
    def test_retries_tool_use_failed_then_succeeds(self, monkeypatch):
        import io
        import time as _time
        import urllib.error
        import urllib.request

        from pynerve import Agent, AgentConfig

        agent = Agent(nv=FakeNv(), config=AgentConfig(max_steps=2))
        agent._tool_map = {}
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.HTTPError(
                    req.full_url, 400, "Bad Request", {},
                    io.BytesIO(b'{"error":{"code":"tool_use_failed"}}'),
                )
            return io.BytesIO(b'{"choices":[{"message":{"content":"ok","tool_calls":[]}}]}')

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(_time, "sleep", lambda s: None)

        result = agent.run("hi")
        assert result.final_answer == "ok"
        assert calls["n"] == 2  # exactly one retry

    def test_no_retry_on_auth_error(self, monkeypatch):
        import io
        import urllib.error
        import urllib.request

        from pynerve import Agent, AgentConfig, AgentError

        agent = Agent(nv=FakeNv(), config=AgentConfig(max_steps=2))
        agent._tool_map = {}
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            raise urllib.error.HTTPError(
                req.full_url, 401, "Unauthorized", {},
                io.BytesIO(b'{"error":{"message":"Invalid API Key"}}'),
            )

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(AgentError):
            agent.run("hi")
        assert calls["n"] == 1  # auth errors are not retried


class TestContentActionParsing:
    def test_plain_json_object(self):
        parsed = Agent._parse_content_action('{"name": "click", "arguments": {"text": "File"}}')
        assert parsed == {
            "name": "click",
            "arguments": {"text": "File"},
        }

    def test_tool_alias(self):
        parsed = Agent._parse_content_action('{"tool": "press_key", "arguments": {"key": "enter"}}')
        assert parsed == {"name": "press_key", "arguments": {"key": "enter"}}

    def test_fenced_json(self):
        text = '```json\n{"name": "click", "arguments": {"text": "Save"}}\n```'
        assert Agent._parse_content_action(text)["name"] == "click"

    def test_plain_text_is_none(self):
        assert Agent._parse_content_action("I cannot do that.") is None

    def test_multiple_fenced_blocks(self):
        text = (
            '```json\n{"name": "launch", "arguments": {"app": "chrome"}}\n```\n'
            '```json\n{"name": "press_key", "arguments": {"key": "enter"}}\n```'
        )
        parsed = Agent._parse_content_action(text)
        assert parsed == {"name": "launch", "arguments": {"app": "chrome"}}

    def test_json_embedded_in_prose(self):
        text = 'I will do it now: {"tool": "click", "arguments": {"text": "OK"}} and then verify.'
        assert Agent._parse_content_action(text) == {"name": "click", "arguments": {"text": "OK"}}

    def test_empty_arguments_object(self):
        # Empty {} must not fall back to treating the whole dict as arguments.
        text = '{"name": "observe", "arguments": {}}'
        assert Agent._parse_content_action(text) == {"name": "observe", "arguments": {}}

    def test_xml_style_action_tags(self):
        # llama-family models often emit <launch>{"app": "chrome"};</launch>
        text = (
            '<launch>{"app": "chrome"};</launch>\n'
            '<type_into>{"text": "search", "content": "lofi"}</type_into>'
        )
        actions = Agent._parse_content_actions(text)
        assert actions == [
            {"name": "launch", "arguments": {"app": "chrome"}},
            {"name": "type_into", "arguments": {"text": "search", "content": "lofi"}},
        ]

    def test_multiple_actions_all_executed(self):
        fake = FakeNv()
        responses = [
            {
                "choices": [
                    _message(
                        '```json\n{"name": "launch", "arguments": {"app": "chrome"}}\n```\n'
                        '```json\n{"name": "press_key", "arguments": {"key": "enter"}}\n```'
                    )
                ]
            },
            {"choices": [_message("Chrome is up", [])]},
        ]
        agent, _ = _make_agent(fake, responses)
        result = agent.run("open chrome")
        assert result.success
        assert fake.calls == [("launch", "chrome"), ("press_key", "enter")]
        assert result.transcript[0]["step"] == result.transcript[1]["step"]  # same LLM turn


class TestCompletionTools:
    def test_done_tool_terminates_immediately(self):
        fake = FakeNv()
        responses = [
            {"choices": [_message("", [_tool_call("done", {"summary": "Completed YouTube search"})])]},
        ]
        agent, _ = _make_agent(fake, responses)
        result = agent.run("search youtube")
        assert result.success is True
        assert result.final_answer == "Completed YouTube search"
        assert result.steps == 1

    def test_fail_tool_returns_failure(self):
        fake = FakeNv()
        responses = [
            {"choices": [_message("", [_tool_call("fail", {"reason": "App not installed"})])]},
        ]
        agent, _ = _make_agent(fake, responses)
        result = agent.run("open nonexistent app")
        assert result.success is False
        assert result.final_answer == "App not installed"


class TestObserveFormattingAndDiff:
    def test_filter_noise(self):
        from pynerve.agent import _filter_noise

        elements = [
            {"text": "File", "confidence": 0.95, "center": [10, 10]},
            {"text": "a", "confidence": 0.95, "center": [20, 20]},  # valid single char -> kept
            {"text": "1", "confidence": 0.95, "center": [30, 30]},  # valid button digit -> kept
            {"text": "Edit", "confidence": 0.1, "center": [40, 40]},  # low conf -> dropped
            {"text": "   ", "confidence": 0.9, "center": [45, 45]},  # empty -> dropped
            {"text": "Save As", "confidence": 0.88, "center": [50, 50]},
        ]
        filtered = _filter_noise(elements)
        assert [e["text"] for e in filtered] == ["File", "a", "1", "Save As"]


    def test_format_observe_row_grouping(self):
        from pynerve.agent import _format_observe

        elements = [
            {"text": "File", "confidence": 0.99, "center": [10.0, 12.0], "bounds": [0, 0, 20, 20]},
            {"text": "Edit", "confidence": 0.99, "center": [50.0, 15.0], "bounds": [40, 0, 60, 20]},
            {"text": "Content", "confidence": 0.95, "center": [10.0, 200.0], "bounds": [0, 190, 50, 210]},
        ]
        formatted = _format_observe(elements)
        assert "row y≈" in formatted
        assert "'File' | 'Edit'" in formatted
        assert "'Content'" in formatted

    def test_compute_observe_diff(self):
        from pynerve.agent import _compute_observe_diff

        before = [{"text": "File"}, {"text": "Edit"}]
        after = [{"text": "File"}, {"text": "Edit"}, {"text": "Save As..."}]
        diff = _compute_observe_diff(before, after)
        assert "NEW on screen: ['save as...']" in diff

        # Screen unchanged
        diff_same = _compute_observe_diff(after, after)
        assert "(screen unchanged)" in diff_same


class TestContextCompression:
    def test_compress_history(self):
        messages = [
            {"role": "system", "content": "You are Py-Nerve."},
            {"role": "user", "content": "Task: test"},
        ]
        # Add 15 turns
        for i in range(15):
            messages.append({"role": "assistant", "content": f"thought {i}", "tool_calls": [{"id": f"c{i}"}]})
            messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"result {i}"})

        compressed = Agent._compress_history(messages, max_messages=8)
        assert len(compressed) <= 9
        assert compressed[0]["role"] == "system"
        assert compressed[1]["content"] == "Task: test"
        # Summary notice injected
        assert "Earlier in this session" in compressed[2]["content"]


class TestConsecutiveErrorRecovery:
    def test_recovery_guidance_injected_after_errors(self):
        fake = FakeNv()
        # 3 errors in a row then success
        responses = [
            {"choices": [_message("", [_tool_call("invalid_tool", {})])]},
            {"choices": [_message("", [_tool_call("invalid_tool", {})])]},
            {"choices": [_message("", [_tool_call("invalid_tool", {})])]},
            {"choices": [_message("", [_tool_call("done", {"summary": "recovered"})])]},
        ]
        agent, seen = _make_agent(fake, responses, max_steps=6)
        result = agent.run("do task")
        assert result.success is True
        # Check that recovery guidance was injected in messages seen by turn 4
        last_turn_messages = seen[-1]
        assert any("consecutive errors" in m.get("content", "") for m in last_turn_messages if m.get("role") == "user")


class TestSpecializedTools:
    def test_detect_dialog_helper(self):
        from pynerve.agent import _detect_dialog

        elements_with_dialog = [
            {"text": "Save Document?"},
            {"text": "Save"},
            {"text": "Don't Save"},
            {"text": "Cancel"},
        ]
        dialog = _detect_dialog(elements_with_dialog)
        assert dialog is not None
        assert dialog["type"] == "save_confirmation"
        assert "Cancel" in dialog["buttons"]

        no_dialog = [{"text": "Hello"}, {"text": "World"}]
        assert _detect_dialog(no_dialog) is None

