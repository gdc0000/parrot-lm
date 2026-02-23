"""Input validation helpers and API-key resolution for the orchestration pipeline."""

from __future__ import annotations

import os
from typing import Any, Dict

from dotenv import load_dotenv
import streamlit as st


def get_openrouter_api_key() -> str:
    """Resolve the OpenRouter API key from session state, environment, or .env file."""
    try:
        api_key = st.session_state.get("openrouter_api_key")
    except Exception:
        api_key = None
    if api_key:
        return api_key

    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        return api_key

    # Load .env lazily so normal env-based deployments do not pay this cost on every import.
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables or .env file.")
    return api_key


def validate_non_empty_string(value: Any, field_name: str) -> str:
    """Validate that a value is a non-empty string and return the stripped result."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{field_name}` must be a non-empty string.")
    return value.strip()


def validate_positive_int(value: Any, field_name: str, default: int) -> int:
    """Validate an optional positive integer value with fallback default."""
    resolved = default if value is None else value
    if not isinstance(resolved, int) or resolved <= 0:
        raise ValueError(f"`{field_name}` must be a positive integer.")
    return resolved


def validate_generation_params(params: Any, field_name: str) -> Dict[str, Any]:
    """Validate optional per-agent model generation parameters."""
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise TypeError(f"`{field_name}` must be a dictionary.")
    return params


def normalize_response_data(response_data: Any) -> Dict[str, Any]:
    """Validate and normalize one agent response payload."""
    if not isinstance(response_data, dict):
        raise TypeError("`response_data` must be a dictionary.")

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
        missing_csv = ", ".join(missing_fields)
        raise KeyError(f"Missing response fields: {missing_csv}")

    content_value = str(response_data["content"] or "").strip()
    return {
        "content": content_value,
        "latency_ms": float(response_data["latency_ms"]),
        "input_tokens": int(response_data["input_tokens"]),
        "output_tokens": int(response_data["output_tokens"]),
        "finish_reason": str(response_data["finish_reason"] or "unknown"),
        "is_refusal": bool(response_data["is_refusal"]),
    }
