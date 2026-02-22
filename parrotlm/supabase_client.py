"""Supabase client singleton for ParrotLM."""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_client: Optional[object] = None
_initialised = False


def get_supabase_client():
    """Return a shared Supabase client, or *None* when credentials are missing.

    The client is created lazily on first call and cached for the process
    lifetime.  Import errors (missing ``supabase`` package) or absent
    environment variables are handled gracefully — a warning is logged and
    ``None`` is returned so the rest of the app keeps working.
    """
    global _client, _initialised  # noqa: PLW0603

    if _initialised:
        return _client

    _initialised = True

    # Ensure .env is loaded
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_ANON_KEY", "").strip()

    if not url or not key:
        logger.warning(
            "supabase_credentials_missing | "
            "Set SUPABASE_URL and SUPABASE_ANON_KEY in your .env file to enable cloud logging."
        )
        return None

    try:
        from supabase import create_client  # type: ignore[import-untyped]

        _client = create_client(url, key)
        logger.info("supabase_client_created | url=%s", url)
    except Exception:
        logger.exception("supabase_client_creation_failed")
        _client = None

    return _client


def reset_client() -> None:
    """Reset the cached client.  Useful for testing."""
    global _client, _initialised  # noqa: PLW0603
    _client = None
    _initialised = False
