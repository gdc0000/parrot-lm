"""Supabase client singleton for ParrotLM."""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_client: Optional[object] = None
_initialised = False


def get_supabase_client(url: Optional[str] = None, key: Optional[str] = None):
    """Return a shared Supabase client, or *None* when credentials are missing.

    The client is created lazily on first call and cached for the process
    lifetime. If *url* and *key* are provided, they override environment
    variables (which is preferred for single-source-of-truth config).
    """
    global _client, _initialised  # noqa: PLW0603

    if _initialised and url is None and key is None:
        return _client

    effective_url = url or os.getenv("SUPABASE_URL", "").strip()
    effective_key = key or os.getenv("SUPABASE_ANON_KEY", "").strip()

    if not effective_url or not effective_key:
        if not _initialised:
            logger.warning(
                "supabase_credentials_missing | "
                "Set SUPABASE_URL and SUPABASE_ANON_KEY in your .env file to enable cloud logging."
            )
        return None

    try:
        from supabase import create_client  # type: ignore[import-untyped]

        _client = create_client(effective_url, effective_key)
        _initialised = True
        logger.info("supabase_client_created | url=%s", effective_url)
    except Exception:
        logger.exception("supabase_client_creation_failed")
        if not _initialised:
            _client = None

    return _client


def reset_client() -> None:
    """Reset the cached client.  Useful for testing."""
    global _client, _initialised  # noqa: PLW0603
    _client = None
    _initialised = False
