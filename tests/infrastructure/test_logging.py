import json
import logging

from parrotlm.infrastructure._logging import (
    parse_message_event,
    extract_record_attributes,
    is_retryable_exception,
    log_structured,
    HumanReadableFormatter,
    JsonLineFormatter,
)


def test_parse_message_event_happy_path():
    event, context = parse_message_event('my_event | {"key": "value"}')
    assert event == "my_event"
    assert context == {"key": "value"}


def test_parse_message_event_failure():
    event, context = parse_message_event("my_event | not_json")
    assert event == "my_event"
    assert context == {"message": "not_json"}


def test_extract_record_attributes():
    record = logging.LogRecord("name", logging.INFO, "path", 1, "msg", (), None)
    record.custom_attr = "custom_value"
    context = {}
    extract_record_attributes(record, context)
    assert context["custom_attr"] == "custom_value"
    assert "name" not in context


def test_is_retryable_exception():
    assert is_retryable_exception(Exception()) is True
    assert is_retryable_exception(ValueError()) is False
    assert is_retryable_exception(TypeError()) is False


def test_log_structured_happy_path(caplog):
    with caplog.at_level(logging.INFO):
        log_structured(logging.INFO, "test_event", key="value")
    assert "test_event" in caplog.text
    assert "value" in caplog.text


def test_log_structured_failure(caplog):
    class Unserializable:
        pass

    with caplog.at_level(logging.INFO):
        log_structured(logging.INFO, "test_event", key=Unserializable())
    assert "test_event" in caplog.text
    # Should fallback to str() representation
    assert "Unserializable object" in caplog.text or "key" in caplog.text


def test_formatters():
    record = logging.LogRecord(
        "name", logging.INFO, "path", 1, 'event | {"k": "v"}', (), None
    )
    record.created = 1000000000.0

    hrf = HumanReadableFormatter()
    hrf_str = hrf.format(record)
    assert "INFO | event" in hrf_str

    jlf = JsonLineFormatter()
    jlf_str = jlf.format(record)
    data = json.loads(jlf_str)
    assert data["level"] == "INFO"
    assert data["event"] == "event"
    assert data["k"] == "v"
