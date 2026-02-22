from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from parrotlm.ui import chat_setup_tab
from parrotlm.ui.sidebar import TechnicalSettings


class _NoopSpinner:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_build_scenario_name_truncates_personas():
    name = chat_setup_tab._build_scenario_name(
        "A" * 30,
        "B" * 30,
    )

    assert name == f"{'A' * 15} vs {'B' * 15}"


def test_execute_simulation_returns_logs_on_success():
    fake_streamlit = SimpleNamespace(
        success=lambda _message: None,
        error=lambda _message: None,
        info=lambda _message: None,
    )
    fake_orchestrator = SimpleNamespace(logs=[{"content": "ok"}])
    settings = TechnicalSettings(num_turns=1, temp_a=1.0, temp_b=1.0, max_tokens=1000, context_window=20)
    chat_inputs = chat_setup_tab.ChatSetupInputs(
        model_a_slug="fake/a",
        persona_a="Persona A",
        model_b_slug="fake/b",
        persona_b="Persona B",
        initial_message="Hi",
    )

    with patch.object(chat_setup_tab, "st", fake_streamlit):
        with patch.object(chat_setup_tab, "_create_orchestrator", return_value=fake_orchestrator):
            with patch.object(chat_setup_tab, "_stream_simulation_messages", return_value=12):
                result = chat_setup_tab._execute_simulation(settings, chat_inputs, chat_container=object())

    assert result == [{"content": "ok"}]


def test_execute_simulation_returns_none_on_failure():
    calls = {"error": [], "info": []}

    def _capture_error(message):
        calls["error"].append(message)

    def _capture_info(message):
        calls["info"].append(message)

    fake_streamlit = SimpleNamespace(
        success=lambda _message: None,
        error=_capture_error,
        info=_capture_info,
    )
    settings = TechnicalSettings(num_turns=1, temp_a=1.0, temp_b=1.0, max_tokens=1000, context_window=20)
    chat_inputs = chat_setup_tab.ChatSetupInputs(
        model_a_slug="fake/a",
        persona_a="Persona A",
        model_b_slug="fake/b",
        persona_b="Persona B",
        initial_message="Hi",
    )

    with patch.object(chat_setup_tab, "st", fake_streamlit):
        with patch.object(chat_setup_tab, "_create_orchestrator", side_effect=RuntimeError("api down")):
            result = chat_setup_tab._execute_simulation(settings, chat_inputs, chat_container=object())

    assert result is None
    assert len(calls["error"]) == 1
    assert len(calls["info"]) == 1


def test_persist_simulation_logs_converts_to_dataframe():
    captured = {}

    def _capture_append(_storage, df):
        captured["df"] = df

    fake_streamlit = SimpleNamespace(success=lambda _message: None, warning=lambda _message: None)
    simulation_logs = [{"content": "a"}, {"content": "b"}]

    with patch.object(chat_setup_tab, "st", fake_streamlit):
        with patch.object(chat_setup_tab, "append_and_persist_logs", side_effect=_capture_append):
            with patch("parrotlm.supabase_logger.upload_session_logs", return_value=True):
                chat_setup_tab._persist_simulation_logs(local_storage=object(), simulation_logs=simulation_logs)

    assert isinstance(captured["df"], pd.DataFrame)
    assert len(captured["df"]) == 2


def test_stream_simulation_messages_tolerates_malformed_log_entries():
    logs = [
        {"output_tokens": "not-an-int", "turn_id": 0},
        {"output_tokens": 3, "speaker_model": "fake/a", "content": "ok", "latency_ms": 10, "turn_id": 1},
    ]
    fake_orchestrator = SimpleNamespace(run_simulation=lambda *_args, **_kwargs: iter(logs))
    fake_streamlit = SimpleNamespace(spinner=lambda _message: _NoopSpinner())

    with patch.object(chat_setup_tab, "st", fake_streamlit):
        with patch.object(chat_setup_tab, "_render_chat_message", side_effect=[KeyError("bad row"), None]):
            total_tokens = chat_setup_tab._stream_simulation_messages(
                orchestrator=fake_orchestrator,
                num_turns=2,
                initial_message="Hi",
                model_a_slug="fake/a",
                persona_a="Persona A",
                persona_b="Persona B",
                chat_container=object(),
            )

    assert total_tokens == 3
