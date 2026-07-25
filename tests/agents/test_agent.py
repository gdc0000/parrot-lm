from types import SimpleNamespace
from unittest import mock

import pytest

from parrotlm.agents.agent import Agent


def _chunk(content=None, finish_reason=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


def _stream(*parts, finish_reason="stop", usage=None):
    """Build a fake streaming response: content chunks, finish chunk, usage chunk."""
    chunks = [_chunk(content=part) for part in parts]
    chunks.append(_chunk(finish_reason=finish_reason))
    chunks.append(SimpleNamespace(choices=[], usage=usage))
    return iter(chunks)


def test_agent_init_happy_path():
    agent = Agent("model", "system", "name", "api_key", 10)
    assert agent.model_slug == "model"
    assert agent.max_history_turns == 10


def test_agent_init_failure():
    with pytest.raises(ValueError):
        Agent("", "system", "name", "key", 10)


@mock.patch("parrotlm.agents.agent.time.time", side_effect=[100.0, 101.0])
def test_generate_response_happy_path(mock_time):
    agent = Agent("model", "system", "name", "api_key", 10)

    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=2)
    create_mock = mock.MagicMock(return_value=_stream("hel", "lo", usage=usage))
    agent.client.chat.completions.create = create_mock

    result = agent.generate_response("hi")

    assert result["content"] == "hello"
    assert result["latency_ms"] == 1000.0
    assert result["input_tokens"] == 5
    assert result["output_tokens"] == 2
    assert result["finish_reason"] == "stop"
    assert result["is_refusal"] is False

    # The request must be a streaming request that asks for usage data.
    _, kwargs = create_mock.call_args
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}


def test_generate_response_invokes_on_token_per_chunk():
    agent = Agent("model", "system", "name", "api_key", 10)

    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=3)
    agent.client.chat.completions.create = mock.MagicMock(
        return_value=_stream("one", " ", "two", usage=usage)
    )

    tokens = []
    result = agent.generate_response("hi", on_token=tokens.append)

    assert tokens == ["one", " ", "two"]
    assert result["content"] == "one two"


def test_generate_response_missing_usage_records_zero_tokens(caplog):
    agent = Agent("model", "system", "name", "api_key", 10)

    agent.client.chat.completions.create = mock.MagicMock(
        return_value=_stream("hello", usage=None)
    )

    with caplog.at_level("WARNING", logger="parrotlm.agents.agent"):
        result = agent.generate_response("hi")

    assert result["content"] == "hello"
    assert result["input_tokens"] == 0
    assert result["output_tokens"] == 0
    assert any("no usage data" in record.getMessage() for record in caplog.records)


def test_execute_api_request_failure():
    agent = Agent("model", "system", "name", "api_key", 10)
    agent.client.chat.completions.create = mock.MagicMock(
        side_effect=ValueError("API Error")
    )

    # ValueError is not retryable (bad caller input), so this fails immediately.
    with pytest.raises(ValueError, match="API Error"):
        agent.execute_api_request([{"role": "user", "content": "hi"}], {})


@mock.patch("tenacity.nap.time.sleep")  # To speed up tenacity
def test_generate_response_tenacity_exhaustion(mock_sleep):
    agent = Agent("model", "system", "name", "api_key", 10)

    # Stream *creation* failures are retried (3 attempts), then re-raised.
    create_mock = mock.MagicMock(side_effect=RuntimeError("Transient error"))
    agent.client.chat.completions.create = create_mock

    with pytest.raises(RuntimeError, match="Transient error"):
        agent.generate_response("hi")

    assert create_mock.call_count == 3
