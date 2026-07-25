"""Supabase client singleton for ParrotLM."""

from __future__ import annotations

import logging
import os
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# We cache the Supabase client globally at the module level because creating it
# involves setting up HTTP connection pools. Reusing the client prevents connection
# exhaustion and improves performance across multiple database operations.
_client: Optional[object] = None
_is_initialized = False


def _read_streamlit_secret(name: str) -> str:
    """Read a value from Streamlit's secrets.toml, if running under Streamlit.

    Returns an empty string when Streamlit is unavailable, no secrets file
    exists, or the key is missing.
    """
    try:
        import streamlit as st

        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def _read_secrets_file(name: str) -> str:
    """Read a top-level string value from ./.streamlit/secrets.toml directly.

    This lets the same secrets.toml used on Streamlit Cloud work on localhost
    without depending on the Streamlit runtime. Uses tomllib when available
    (Python >= 3.11) and falls back to a flat `KEY = "value"` line parse.
    """
    secrets_path = os.path.join(os.getcwd(), ".streamlit", "secrets.toml")
    if not os.path.isfile(secrets_path):
        return ""

    try:
        try:
            import tomllib

            with open(secrets_path, "rb") as handle:
                data = tomllib.load(handle)
            value = data.get(name, "")
            return str(value).strip() if isinstance(value, str) else ""
        except ImportError:
            pass

        pattern = re.compile(
            rf'^\s*{re.escape(name)}\s*=\s*["\']([^"\']*)["\']',
            re.MULTILINE,
        )
        with open(secrets_path, "r", encoding="utf-8") as handle:
            match = pattern.search(handle.read())
        return match.group(1).strip() if match else ""
    except Exception:
        return ""


def resolve_supabase_credentials(
    provided_url: Optional[str], provided_key: Optional[str]
) -> Tuple[str, str]:
    """Resolve the final Supabase URL and key to use.

    Resolution order: explicitly provided values, then environment variables
    (.env), then Streamlit secrets (runtime), then ./.streamlit/secrets.toml.

    Args:
        provided_url: An optional URL provided directly by the caller.
        provided_key: An optional key provided directly by the caller.

    Returns:
        A tuple containing the resolved URL and key strings.
    """
    effective_url = provided_url or os.getenv("SUPABASE_URL", "").strip()
    effective_key = provided_key or os.getenv("SUPABASE_ANON_KEY", "").strip()
    if not effective_url:
        effective_url = _read_streamlit_secret("SUPABASE_URL")
    if not effective_key:
        effective_key = _read_streamlit_secret("SUPABASE_ANON_KEY")
    if not effective_url:
        effective_url = _read_secrets_file("SUPABASE_URL")
    if not effective_key:
        effective_key = _read_secrets_file("SUPABASE_ANON_KEY")
    return effective_url, effective_key


def instantiate_supabase_client(url: str, key: str) -> Optional[object]:
    """Create a new Supabase client instance and update global state.

    Args:
        url: The validated Supabase project URL.
        key: The validated Supabase anonymous API key.

    Returns:
        The instantiated Supabase client object, or None if creation failed.
    """
    global _client, _is_initialized  # noqa: PLW0603

    try:
        from supabase import create_client  # type: ignore[import-untyped]

        _client = create_client(url, key)
        _is_initialized = True
        logger.info("supabase_client_created | url=%s", url)
    except Exception as exception:
        logger.exception("supabase_client_creation_failed | reason=%s", str(exception))
        if not _is_initialized:
            _client = None

    return _client


def get_supabase_client(
    url: Optional[str] = None, key: Optional[str] = None
) -> Optional[object]:
    """Return a shared Supabase client, or None when credentials are missing.

    The client is created lazily on first call and cached for the process
    lifetime. If `url` and `key` are provided, they override environment
    variables (which is preferred for single-source-of-truth config).

    Args:
        url: Optional Supabase URL to use instead of environment variables.
        key: Optional Supabase anon key to use instead of environment variables.

    Returns:
        The cached or newly created Supabase client object, or None if credentials
        are missing or if client creation failed.
    """
    global _client, _is_initialized  # noqa: PLW0603

    if _is_initialized and url is None and key is None:
        return _client

    effective_url, effective_key = resolve_supabase_credentials(url, key)

    if not effective_url or not effective_key:
        if not _is_initialized:
            logger.warning(
                "supabase_credentials_missing | "
                "Set SUPABASE_URL and SUPABASE_ANON_KEY in your .env file to enable cloud logging."
            )
            # Cache the miss so subsequent callers don't re-warn. The cached-None
            # early return above serves all later calls until reset_client().
            _is_initialized = True
        return None

    return instantiate_supabase_client(effective_url, effective_key)


def reset_client() -> None:
    """Reset the cached client state.

    This function is primarily useful for testing to ensure a clean state
    between test runs without test pollution.
    """
    global _client, _is_initialized  # noqa: PLW0603
    _client = None
    _is_initialized = False
