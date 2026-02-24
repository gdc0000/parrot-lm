import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import mock_open, patch

import pytest

from parrotlm.agent import Agent
from parrotlm.orchestrator import AgentConfig, Orchestrator


def _agent_config(
    model: str,
    system_prompt: str,
    user_persona_snapshot: str = "",
    max_history_turns: int = 20,
    parameters: Any = None,
) -> AgentConfig:
    return AgentConfig(
        model=model,
        system_prompt=system_prompt,
        user_persona_snapshot=user_persona_snapshot,
        max_history_turns=max_history_turns,
        parameters={} if parameters is None else parameters,
    )


class _FakeCompletions:
    def create(self, model, messages, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="mocked reply"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


class _FakeOpenAIClient:
    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


class _RecordingCompletions:
    def __init__(self):
        self.calls = []

    def create(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="mocked reply"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


class _RecordingOpenAIClient:
    def __init__(self, *args, **kwargs):
        self._completions = _RecordingCompletions()
        self.chat = SimpleNamespace(completions=self._completions)


class _EmptyContentCompletions:
    def create(self, model, messages, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="   "),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=0),
        )


class _EmptyContentOpenAIClient:
    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=_EmptyContentCompletions())


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False)
@patch("parrotlm.agent.OpenAI", side_effect=_FakeOpenAIClient)
def test_agent_generate_response_returns_expected_fields(_mock_openai):
    agent = Agent(
        model_slug="fake/model",
        system_prompt="You are concise.",
        name="Agent A",
        api_key="test-api-key",
        max_history_turns=5,
    )

    response = agent.generate_response("hello")

    assert response["content"] == "mocked reply"
    assert response["finish_reason"] == "stop"
    assert response["input_tokens"] == 10
    assert response["output_tokens"] == 5
    assert response["is_refusal"] is False


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False)
@patch("parrotlm.agent.OpenAI", side_effect=_FakeOpenAIClient)
def test_run_simulation_emits_two_entries_for_one_turn(_mock_openai):
    agent_a_config = _agent_config(
        "fake/model-a", "Persona A", user_persona_snapshot="Persona A"
    )
    agent_b_config = _agent_config(
        "fake/model-b", "Persona B", user_persona_snapshot="Persona B"
    )

    orchestrator = Orchestrator(
        agent_a_config,
        agent_b_config,
        scenario_name="test",
        openrouter_api_key="test-key",
    )
    logs = list(orchestrator.run_simulation(num_turns=1, initial_message="Hi"))

    assert len(logs) == 2
    assert "content" in logs[0]
    assert "system_prompt_snapshot" in logs[0]


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False)
@patch("parrotlm.agent.OpenAI", side_effect=_FakeOpenAIClient)
def test_orchestrator_uses_distinct_history_windows_per_agent(_mock_openai):
    agent_a_config = _agent_config("fake/model-a", "Persona A", max_history_turns=3)
    agent_b_config = _agent_config("fake/model-b", "Persona B", max_history_turns=7)

    orchestrator = Orchestrator(
        agent_a_config,
        agent_b_config,
        scenario_name="test",
        openrouter_api_key="test-key",
    )

    assert orchestrator.agent_a.max_history_turns == 3
    assert orchestrator.agent_b.max_history_turns == 7


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False)
@patch("parrotlm.agent.OpenAI", side_effect=_FakeOpenAIClient)
def test_orchestrator_rejects_non_dict_agent_params(_mock_openai):
    agent_a_config = _agent_config("fake/model-a", "Persona A", parameters=["invalid"])
    agent_b_config = _agent_config("fake/model-b", "Persona B")

    with pytest.raises(TypeError):
        Orchestrator(
            agent_a_config,
            agent_b_config,
            scenario_name="test",
            openrouter_api_key="test-key",
        )


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False)
@patch("parrotlm.agent.OpenAI", side_effect=_EmptyContentOpenAIClient)
def test_agent_generate_response_marks_empty_content_as_refusal(_mock_openai):
    agent = Agent(
        model_slug="fake/model",
        system_prompt="You are concise.",
        name="Agent A",
        api_key="test-key",
        max_history_turns=5,
    )

    response = agent.generate_response("hello")

    assert response["content"] == ""
    assert response["is_refusal"] is True
    assert response["output_tokens"] == 0
    assert (
        len(agent.history) == 2
    )  # system + user; assistant is not appended on blank content.


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False)
@patch("parrotlm.agent.OpenAI", side_effect=_RecordingOpenAIClient)
def test_agent_context_window_never_starts_with_assistant_after_trimming(_mock_openai):
    agent = Agent(
        model_slug="fake/model",
        system_prompt="You are concise.",
        name="Agent A",
        api_key="test-key",
        max_history_turns=1,
    )

    agent.generate_response("first")
    agent.generate_response("second")

    recording_client = getattr(agent, "client", agent.client)
    completions: Any = recording_client.chat.completions
    second_call_messages = completions.calls[1]["messages"]
    sent_roles = [message["role"] for message in second_call_messages]

    assert sent_roles[0] == "system"
    assert sent_roles[1:] == ["user"]


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False)
@patch("parrotlm.agent.OpenAI", side_effect=_FakeOpenAIClient)
def test_run_simulation_stops_after_agent_a_refusal(_mock_openai):
    agent_a_config = _agent_config("fake/model-a", "Persona A")
    agent_b_config = _agent_config("fake/model-b", "Persona B")
    orchestrator = Orchestrator(
        agent_a_config,
        agent_b_config,
        scenario_name="test",
        openrouter_api_key="test-key",
    )

    refusal_response = {
        "content": "",
        "latency_ms": 1.0,
        "input_tokens": 1,
        "output_tokens": 0,
        "finish_reason": "stop",
        "is_refusal": True,
    }

    with patch.object(
        orchestrator.agent_a, "generate_response", return_value=refusal_response
    ):
        with patch.object(
            orchestrator.agent_b, "generate_response"
        ) as mock_agent_b_generate:
            logs = list(orchestrator.run_simulation(num_turns=3, initial_message="Hi"))

    assert len(logs) == 1
    mock_agent_b_generate.assert_not_called()


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False)
@patch("parrotlm.agent.OpenAI", side_effect=_FakeOpenAIClient)
def test_run_simulation_wraps_agent_failure(_mock_openai):
    agent_a_config = _agent_config("fake/model-a", "Persona A")
    agent_b_config = _agent_config("fake/model-b", "Persona B")
    orchestrator = Orchestrator(
        agent_a_config,
        agent_b_config,
        scenario_name="test",
        openrouter_api_key="test-key",
    )

    with patch.object(
        orchestrator.agent_a,
        "generate_response",
        side_effect=RuntimeError("api failure"),
    ):
        with pytest.raises(RuntimeError) as raised:
            list(orchestrator.run_simulation(num_turns=1, initial_message="Hi"))

    assert "Agent A failed on turn 0." in str(raised.value)
    assert "api failure" in str(raised.value.__cause__)


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False)
@patch("parrotlm.agent.OpenAI", side_effect=_FakeOpenAIClient)
def test_run_simulation_wraps_invalid_agent_payload(_mock_openai):
    agent_a_config = _agent_config("fake/model-a", "Persona A")
    agent_b_config = _agent_config("fake/model-b", "Persona B")
    orchestrator = Orchestrator(
        agent_a_config,
        agent_b_config,
        scenario_name="test",
        openrouter_api_key="test-key",
    )

    invalid_payload = {
        "content": "ok",
        "latency_ms": 1.0,
        # missing token and finish fields on purpose
    }

    with patch.object(
        orchestrator.agent_a, "generate_response", return_value=invalid_payload
    ):
        with pytest.raises(RuntimeError) as raised:
            list(orchestrator.run_simulation(num_turns=1, initial_message="Hi"))

    assert "returned an invalid payload" in str(raised.value)
