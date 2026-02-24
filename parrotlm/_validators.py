from __future__ import annotations

from typing import Any, Dict


def validate_non_empty_string(value: Any, field_name: str) -> str:
    """Validate that a value is a non-empty string and return the stripped result.

    Args:
        value: The string to validate.
        field_name: The name of the field being validated.

    Returns:
        The stripped string.

    Raises:
        ValueError: If the value is not a string or if the stripped string is empty. Empty strings cause critical failures downstream when used as identifiers or prompts.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"`{field_name}` must be a non-empty string. Received: {value}"
        )
    return value.strip()


def validate_positive_int(value: Any, field_name: str, default: int) -> int:
    """Validate an optional positive integer value with fallback default.

    Args:
        value: The integer value to validate.
        field_name: The name of the field being validated.
        default: The fallback value if `value` is None.

    Returns:
        The resolved positive integer.

    Raises:
        ValueError: If the resolved value is not an integer or is less than or equal to zero. Non-positive integers cause downstream index out-of-bounds errors.
    """
    resolved = default if value is None else value
    if not isinstance(resolved, int) or resolved <= 0:
        raise ValueError(
            f"`{field_name}` must be a positive integer. Received: {resolved}"
        )
    return resolved


def validate_generation_parameters(parameters: Any, field_name: str) -> Dict[str, Any]:
    """Validate optional per-agent model generation parameters.

    Args:
        parameters: The dictionary of generation parameters to validate.
        field_name: The name of the field being validated.

    Returns:
        The validated parameter dictionary, or an empty dictionary if None was provided.

    Raises:
        TypeError: If `parameters` is provided but is not a dictionary.
    """
    if parameters is None:
        return {}
    if not isinstance(parameters, dict):
        raise TypeError(
            f"`{field_name}` must be a dictionary. Received type: {type(parameters).__name__}"
        )
    return parameters


def verify_required_fields(response_data: Any) -> None:
    """Verify that the response data is a dictionary and contains all required keys.

    Args:
        response_data: The dictionary to check for required fields.

    Raises:
        TypeError: If `response_data` is not a dictionary.
        KeyError: If any required fields are missing from the dictionary.
    """
    if not isinstance(response_data, dict):
        raise TypeError(
            f"`response_data` must be a dictionary. Received type: {type(response_data).__name__}"
        )

    required_fields = [
        "content",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "finish_reason",
        "is_refusal",
    ]
    missing_fields = [field for field in required_fields if field not in response_data]
    if missing_fields:
        missing_comma_separated_values = ", ".join(missing_fields)
        raise KeyError(f"Missing response fields: {missing_comma_separated_values}")


def cast_response_data_types(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """Cast the values in the response data dictionary to their appropriate types.

    Args:
        response_data: A dictionary containing the raw response fields.

    Returns:
        A new dictionary with the cast values.

    Raises:
        ValueError: If a field cannot be cast to its expected type.
        KeyError: If an expected field is missing.
    """
    content_value = str(response_data["content"] or "").strip()
    return {
        "content": content_value,
        "latency_ms": float(response_data["latency_ms"]),
        "input_tokens": int(response_data["input_tokens"]),
        "output_tokens": int(response_data["output_tokens"]),
        "finish_reason": str(response_data["finish_reason"] or "unknown"),
        "is_refusal": bool(response_data["is_refusal"]),
    }


def normalize_response_data(response_data: Any) -> Dict[str, Any]:
    """Validate and normalize one agent response payload.

    Args:
        response_data: The raw dictionary returned by the agent API.

    Returns:
        A normalized dictionary with specific types for all required fields.

    Raises:
        TypeError: If the input is not a dictionary.
        KeyError: If required fields are missing.
        ValueError: If fields cannot be cast to their required types.
    """
    verify_required_fields(response_data)
    return cast_response_data_types(response_data)
