"""Tests for parrotlm.infrastructure.supabase_logger."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from parrotlm.infrastructure import supabase_logger
from parrotlm.infrastructure.supabase_client import reset_client


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

    @patch(
        "parrotlm.infrastructure.supabase_logger.get_supabase_client",
        return_value=None,
    )
    def test_returns_false_when_client_unavailable(self, _mock_client):
        result = supabase_logger.upload_session_logs(_sample_logs())
        assert result[0] is False
        assert isinstance(result[1], str)

    @patch("parrotlm.infrastructure.supabase_logger.get_supabase_client")
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

    @patch("parrotlm.infrastructure.supabase_logger.get_supabase_client")
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

    def test_clean_application_log_entry_strips_unknown_keys(self):
        entry = [{"event": "started", "unknown_key": "discard", "level": "INFO"}]
        cleaned = supabase_logger.sanitize_application_log_entries(entry)
        assert "unknown_key" not in cleaned[0]
        assert cleaned[0]["event"] == "started"
        assert cleaned[0]["level"] == "INFO"


class TestSupabaseLogHandler:
    """Unit tests for SupabaseLogHandler."""

    def test_formats_log_record_for_supabase(self):
        record = logging.LogRecord(
            "parrotlm.test",
            logging.INFO,
            "path.py",
            42,
            'test_event | {"key": "value"}',
            (),
            None,
        )

        row = supabase_logger.format_log_record_for_supabase(record)

        assert row["level"] == "INFO"
        assert row["logger_name"] == "parrotlm.test"
        assert row["event"] == "test_event"
        assert row["context"]["key"] == "value"

    def test_handler_uploads_when_batch_size_reached(self):
        mock_table = MagicMock()
        mock_table.insert.return_value.execute.return_value = SimpleNamespace(
            data=[{"id": 1}, {"id": 2}]
        )
        mock_client = MagicMock()
        mock_client.table.return_value = mock_table
        handler = supabase_logger.SupabaseLogHandler(batch_size=2, client=mock_client)

        record_1 = logging.LogRecord(
            "parrotlm.test", logging.INFO, "path.py", 1, "event_one", (), None
        )
        record_2 = logging.LogRecord(
            "parrotlm.test", logging.WARNING, "path.py", 2, "event_two", (), None
        )

        handler.handle(record_1)
        mock_table.insert.assert_not_called()

        handler.handle(record_2)
        supabase_logger.wait_for_pending_uploads()

        mock_client.table.assert_called_once_with("application_logs")
        inserted_rows = mock_table.insert.call_args[0][0]
        assert [row["event"] for row in inserted_rows] == ["event_one", "event_two"]
        assert handler.buffer == []

    def test_handler_ignores_supabase_infrastructure_logs(self):
        mock_table = MagicMock()
        mock_table.insert.return_value.execute.return_value = SimpleNamespace(data=[])
        mock_client = MagicMock()
        mock_client.table.return_value = mock_table
        handler = supabase_logger.SupabaseLogHandler(batch_size=1, client=mock_client)
        record = logging.LogRecord(
            "parrotlm.infrastructure.supabase_client",
            logging.INFO,
            "path.py",
            1,
            "internal_event",
            (),
            None,
        )

        handler.handle(record)

        mock_table.insert.assert_not_called()

    def test_install_supabase_log_handler_adds_one_root_handler(self):
        root_logger = logging.getLogger()
        original_handlers = list(root_logger.handlers)
        root_logger.handlers = []
        mock_client = MagicMock()
        try:
            handler_1 = supabase_logger.install_supabase_log_handler(
                batch_size=5, client=mock_client
            )
            handler_2 = supabase_logger.install_supabase_log_handler(
                batch_size=5, client=mock_client
            )

            assert handler_1 is handler_2
            assert len(root_logger.handlers) == 1
        finally:
            for handler in root_logger.handlers:
                handler.close()
            root_logger.handlers = original_handlers


class TestSupabaseBufferedLogger:
    """Unit tests for SupabaseBufferedLogger."""

    def setup_method(self):
        reset_client()

    @patch("parrotlm.infrastructure.supabase_logger.get_supabase_client")
    def test_buffer_uploads_on_batch_size(self, mock_get_client):
        mock_table = MagicMock()
        mock_table.insert.return_value.execute.return_value = SimpleNamespace(
            data=[{"id": 1}, {"id": 2}]
        )
        mock_client = MagicMock()
        mock_client.table.return_value = mock_table
        mock_get_client.return_value = mock_client

        logger = supabase_logger.SupabaseBufferedLogger(batch_size=2)

        # First push - should not upload
        logger.push({"turn_id": 0, "content": "first"})
        mock_table.insert.assert_not_called()

        # Second push - should upload
        logger.push({"turn_id": 1, "content": "second"})
        supabase_logger.wait_for_pending_uploads()
        mock_table.insert.assert_called_once()
        assert len(logger.buffer) == 0

    @patch("parrotlm.infrastructure.supabase_logger.get_supabase_client")
    def test_flush_uploads_remaining_logs(self, mock_get_client):
        mock_table = MagicMock()
        mock_table.insert.return_value.execute.return_value = SimpleNamespace(
            data=[{"id": 1}]
        )
        mock_client = MagicMock()
        mock_client.table.return_value = mock_table
        mock_get_client.return_value = mock_client

        logger = supabase_logger.SupabaseBufferedLogger(batch_size=10)
        logger.push({"turn_id": 0, "content": "first"})

        mock_table.insert.assert_not_called()

        logger.flush()
        supabase_logger.wait_for_pending_uploads()
        mock_table.insert.assert_called_once()
        assert len(logger.buffer) == 0


class TestMakeJsonSafe:
    """Unit tests for the direct make_json_safe sanitizer."""

    def test_primitives_pass_through(self):
        assert supabase_logger.make_json_safe("text") == "text"
        assert supabase_logger.make_json_safe(3) == 3
        assert supabase_logger.make_json_safe(1.5) == 1.5
        assert supabase_logger.make_json_safe(True) is True
        assert supabase_logger.make_json_safe(None) is None

    def test_nested_structures_are_sanitized_recursively(self):
        value = {
            "tuple": (1, 2),
            "nested": {"list": [1, {"deep": "x"}]},
            7: "non-string key",
        }
        result = supabase_logger.make_json_safe(value)
        assert result == {
            "tuple": [1, 2],
            "nested": {"list": [1, {"deep": "x"}]},
            "7": "non-string key",
        }

    def test_non_serializable_objects_fall_back_to_str(self):
        class Weird:
            def __str__(self):
                return "weird-instance"

        assert supabase_logger.make_json_safe(Weird()) == "weird-instance"
        assert supabase_logger.make_json_safe({"obj": Weird()}) == {
            "obj": "weird-instance"
        }
