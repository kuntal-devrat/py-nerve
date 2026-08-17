from __future__ import annotations

import json
from unittest.mock import patch

import pynerve as nv
from pynerve.agent import Agent, AgentConfig, build_tools


def test_try_parse_action_stringified_arguments():
    # Test JSON string where arguments is a JSON-encoded string (common in some LLM providers)
    raw_payload = json.dumps({
        "name": "click",
        "arguments": json.dumps({"text": "Submit", "relative_to": "Form"})
    })
    action = Agent._try_parse_action(raw_payload)
    assert action is not None
    assert action["name"] == "click"
    assert action["arguments"] == {"text": "Submit", "relative_to": "Form"}


def test_try_parse_action_dict_arguments():
    raw_payload = json.dumps({
        "name": "type_into",
        "arguments": {"text": "Search", "content": "hello"}
    })
    action = Agent._try_parse_action(raw_payload)
    assert action is not None
    assert action["name"] == "type_into"
    assert action["arguments"] == {"text": "Search", "content": "hello"}


def test_try_parse_action_flat_tool():
    raw_payload = json.dumps({
        "tool": "press_key",
        "key": "enter"
    })
    action = Agent._try_parse_action(raw_payload)
    assert action is not None
    assert action["name"] == "press_key"
    assert action["arguments"] == {"key": "enter"}


def test_build_tools_includes_new_tools():
    tools = build_tools()
    tool_names = {t.name for t in tools}
    assert "hover" in tool_names
    assert "middle_click" in tool_names
    assert "get_clipboard" in tool_names
    assert "set_clipboard" in tool_names
    assert "observe" in tool_names
    assert "click" in tool_names


def test_agent_kwargs_acceptance():
    # Verify step_delay and other options are accepted without TypeError
    cfg = AgentConfig(step_delay=1.5, max_steps=5, dry_run=True)
    assert cfg.step_delay == 1.5

    with patch("pynerve.agent.Agent.run") as mock_run:
        nv.run_agent("Test task", step_delay=2.0, dry_run=True)
        assert mock_run.called
