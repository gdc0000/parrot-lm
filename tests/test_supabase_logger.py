"""Tests for parrotlm.supabase_logger."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from parrotlm import supabase_logger
from parrotlm.supabase_client import reset_client


def _sample_logs():
    return [
        {
            "experiment_id": "exp-001",
            "turn_id": 0,
            "scenario": "test",
            "speaker_model": "model-a",
            "responder_model": "model-b",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "latency_ms": 123.4,
            "input_tokens": 10,
            "output_tokens": 20,
            "content": "Hello!",
            "finish_reason": "stop",
            "is_refusal": False,
            "system_prompt_snapshot": "You are a CTO.",
        }
    ]


class TestUploadSessionLogs:
    """Unit tests for upload_session_logs."""

    def setup_method(self):
        reset_client()

    def teardown_method(self):
        reset_client()

    def test_empty_logs_returns_true(self):
        result = supabase_logger.upload_session_logs([])
        assert result[0] is True
        assert isinstance(result[1], str)

    @patch("parrotlm.supabase_logger.get_supabase_client", return_value=None)
    def test_returns_false_when_client_unavailable(self, _mock_client):
        result = supabase_logger.upload_session_logs(_sample_logs())
        assert result[0] is False
        assert isinstance(result[1], str)

    @patch("parrotlm.supabase_logger.get_supabase_client")
    def test_returns_true_on_successful_insert(self, mock_get_client):
        mock_table = MagicMock()
        mock_table.insert.return_value.execute.return_value = SimpleNamespace(
            data=[{"id": 1}]
        )
        mock_client = MagicMock()
        mock_client.table.return_value = mock_table
        mock_get_client.return_value = mock_client

        result = supabase_logger.upload_session_logs(_sample_logs())

        assert result[0] is True
        assert isinstance(result[1], str)
        mock_client.table.assert_called_once_with("session_logs")
        inserted_rows = mock_table.insert.call_args[0][0]
        assert len(inserted_rows) == 1
        assert inserted_rows[0]["experiment_id"] == "exp-001"

    @patch("parrotlm.supabase_logger.get_supabase_client")
    def test_returns_false_on_insert_exception(self, mock_get_client):
        mock_table = MagicMock()
        mock_table.insert.return_value.execute.side_effect = RuntimeError(
            "network error"
        )
        mock_client = MagicMock()
        mock_client.table.return_value = mock_table
        mock_get_client.return_value = mock_client

        result = supabase_logger.upload_session_logs(_sample_logs())

        assert result[0] is False
        assert isinstance(result[1], str)

    def test_clean_log_entry_strips_unknown_keys(self):
        entry = [{"experiment_id": "e1", "unknown_key": "discard", "turn_id": 0}]
        cleaned = supabase_logger.sanitize_log_entries(entry)
        assert "unknown_key" not in cleaned[0]
        assert cleaned[0]["experiment_id"] == "e1"
        assert cleaned[0]["turn_id"] == 0
