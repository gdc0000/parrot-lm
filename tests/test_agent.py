from unittest import mock
import pytest
from parrotlm.agent import Agent


def test_agent_init_happy_path():
    agent = Agent("model", "system", "name", "api_key", 10)
    assert agent.model_slug == "model"
    assert agent.max_history_turns == 10


def test_agent_init_failure():
    with pytest.raises(ValueError):
        Agent("", "system", "name", "key", 10)


@mock.patch("parrotlm.agent.time.time", side_effect=[100.0, 101.0])
def test_generate_response_happy_path(mock_time):
    agent = Agent("model", "system", "name", "api_key", 10)

    mock_response = mock.MagicMock()
    mock_choice = mock.MagicMock()
    mock_choice.message.content = "hello"
    mock_choice.finish_reason = "stop"
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 5
    mock_response.usage.completion_tokens = 2

    agent.client.chat.completions.create = mock.MagicMock(return_value=mock_response)

    result = agent.generate_response("hi")

    assert result["content"] == "hello"
    assert result["latency_ms"] == 1000.0
    assert result["input_tokens"] == 5


def test_execute_api_request_failure():
    agent = Agent("model", "system", "name", "api_key", 10)
    agent.client.chat.completions.create = mock.MagicMock(
        side_effect=Exception("API Error")
    )

    with pytest.raises(Exception, match="API Error"):
        # The retry decorator is on generate_response, not execute_api_request, so this will fail immediately.
        agent.execute_api_request([{"role": "user", "content": "hi"}], {})


def test_extract_response_metrics_failure():
    agent = Agent("model", "system", "name", "api_key", 10)

    mock_response = mock.MagicMock()
    mock_response.choices = []

    with pytest.raises(
        RuntimeError, match="Model response is missing message choices."
    ):
        agent.extract_response_metrics(mock_response, 100.0)


@mock.patch("parrotlm.agent.time.sleep")  # To speed up tenacity
def test_generate_response_tenacity_exhaustion(mock_sleep):
    agent = Agent("model", "system", "name", "api_key", 10)

    # We patch the inner method that does the API call to fail.
    # We must patch the agent's execute_api_request. Wait, `execute_api_request` is called inside `generate_response`.
    with mock.patch.object(
        agent, "execute_api_request", side_effect=RuntimeError("Transient error")
    ):
        with pytest.raises(RuntimeError, match="Transient error"):
            # Should retry 3 times then fail
            agent.generate_response("hi")
