import pytest
from parrotlm.validation._validators import (
    validate_non_empty_string,
    validate_positive_int,
    validate_generation_parameters,
    verify_required_fields,
    cast_response_data_types,
    normalize_response_data,
)


def test_validate_non_empty_string_happy_path():
    assert validate_non_empty_string(" hello ", "test") == "hello"


def test_validate_non_empty_string_failure():
    with pytest.raises(ValueError, match="must be a non-empty string"):
        validate_non_empty_string("   ", "test")
    with pytest.raises(ValueError, match="must be a non-empty string"):
        validate_non_empty_string(123, "test")


def test_validate_positive_int_happy_path():
    assert validate_positive_int(5, "test", 10) == 5
    assert validate_positive_int(None, "test", 10) == 10


def test_validate_positive_int_failure():
    with pytest.raises(ValueError, match="must be a positive integer"):
        validate_positive_int(0, "test", 10)
    with pytest.raises(ValueError, match="must be a positive integer"):
        validate_positive_int(-5, "test", 10)
    with pytest.raises(ValueError, match="must be a positive integer"):
        validate_positive_int("5", "test", 10)


def test_validate_generation_parameters_happy_path():
    assert validate_generation_parameters(None, "test") == {}
    assert validate_generation_parameters({"temp": 0.5}, "test") == {"temp": 0.5}


def test_validate_generation_parameters_failure():
    with pytest.raises(TypeError, match="must be a dictionary"):
        validate_generation_parameters(["temp"], "test")


def test_verify_required_fields_happy_path():
    data = {
        "content": "hi",
        "latency_ms": 100,
        "input_tokens": 10,
        "output_tokens": 20,
        "finish_reason": "stop",
        "is_refusal": False,
    }
    verify_required_fields(data)  # Should not raise


def test_verify_required_fields_failure():
    with pytest.raises(TypeError, match="must be a dictionary"):
        verify_required_fields("not a dict")

    with pytest.raises(KeyError, match="Missing response fields: content, latency_ms"):
        verify_required_fields(
            {
                "input_tokens": 10,
                "output_tokens": 20,
                "finish_reason": "stop",
                "is_refusal": False,
            }
        )


def test_cast_response_data_types_happy_path():
    data = {
        "content": " hi ",
        "latency_ms": "100.5",
        "input_tokens": "10",
        "output_tokens": "20",
        "finish_reason": None,
        "is_refusal": "",
    }
    result = cast_response_data_types(data)
    assert result == {
        "content": "hi",
        "latency_ms": 100.5,
        "input_tokens": 10,
        "output_tokens": 20,
        "finish_reason": "unknown",
        "is_refusal": False,
    }


def test_normalize_response_data_happy_path():
    data = {
        "content": "hi",
        "latency_ms": 100,
        "input_tokens": 10,
        "output_tokens": 20,
        "finish_reason": "stop",
        "is_refusal": False,
    }
    result = normalize_response_data(data)
    assert result["content"] == "hi"


def test_normalize_response_data_failure():
    with pytest.raises(TypeError):
        normalize_response_data("not dict")
