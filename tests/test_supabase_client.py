import pytest
import os
import sys
from unittest import mock

from parrotlm.infrastructure.supabase_client import (
    get_supabase_client,
    reset_client,
    resolve_supabase_credentials,
    instantiate_supabase_client,
)


@pytest.fixture(autouse=True)
def clean_supabase_client():
    reset_client()
    yield
    reset_client()


def test_resolve_supabase_credentials(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "env_url")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "env_key")

    # Provided overrides env
    url, key = resolve_supabase_credentials("arg_url", "arg_key")
    assert url == "arg_url"
    assert key == "arg_key"

    # Env fallback
    url, key = resolve_supabase_credentials(None, None)
    assert url == "env_url"
    assert key == "env_key"


def test_instantiate_supabase_client_failure(monkeypatch, caplog):
    # Mock create_client to raise an error
    sys.modules["supabase"] = mock.MagicMock()
    sys.modules["supabase"].create_client.side_effect = Exception("Test failure")

    client = instantiate_supabase_client("url", "key")
    assert client is None
    assert "supabase_client_creation_failed | reason=Test failure" in caplog.text


def test_get_supabase_client_missing_credentials(monkeypatch, caplog):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)

    client = get_supabase_client()
    assert client is None
    assert "supabase_credentials_missing" in caplog.text


def test_get_supabase_client_caches_client(monkeypatch):
    # Mock create_client to return a dummy string
    sys.modules["supabase"] = mock.MagicMock()
    sys.modules["supabase"].create_client.return_value = "dummy_client"

    client1 = get_supabase_client("url", "key")
    assert client1 == "dummy_client"

    # Should return cached even without args
    client2 = get_supabase_client()
    assert client2 == "dummy_client"

    sys.modules["supabase"].create_client.assert_called_once()

