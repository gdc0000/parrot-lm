import os
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from parrotlm import analysis_utils
from parrotlm.orchestrator import AgentConfig, Orchestrator
from parrotlm.ui import session_state


class _FakeCompletions:
    def create(self, model, messages, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="mocked reply"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


class _FakeOpenAIClient:
    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


class _FakeLocalStorage:
    def __init__(self):
        self.store = {}

    def setItem(self, key, value):
        self.store[key] = value

    def getItem(self, key):
        return self.store.get(key)


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False)
@patch("parrotlm.agent.OpenAI", side_effect=_FakeOpenAIClient)
def test_orchestration_to_persistence_to_analysis_pipeline(_mock_openai):
    orchestrator = Orchestrator(
        agent_a_config=AgentConfig(
            model="fake/model-a",
            system_prompt="Persona A",
            user_persona_snapshot="Persona A",
            max_history_turns=20,
            params={},
        ),
        agent_b_config=AgentConfig(
            model="fake/model-b",
            system_prompt="Persona B",
            user_persona_snapshot="Persona B",
            max_history_turns=20,
            params={},
        ),
        scenario_name="pipeline-test",
    )

    logs = list(orchestrator.run_simulation(num_turns=1, initial_message="Hi"))
    assert len(logs) == 2

    fake_streamlit = SimpleNamespace(session_state={"all_logs": pd.DataFrame()})
    storage = _FakeLocalStorage()
    with patch.object(session_state, "st", fake_streamlit):
        session_state.append_and_persist_logs(storage, pd.DataFrame(logs))

    assert session_state.LOCAL_STORAGE_LOG_KEY in storage.store
    persisted_logs = storage.store[session_state.LOCAL_STORAGE_LOG_KEY]
    assert len(persisted_logs) == 2
    assert "speaker_model" in persisted_logs[0]

    fake_metrics = {
        "token_count": 2,
        "sentence_count": 1,
        "avg_sentence_length": 2.0,
        "noun_ratio": 0.5,
        "verb_ratio": 0.0,
        "adj_ratio": 0.0,
        "adv_ratio": 0.0,
        "pron_ratio": 0.0,
    }
    with patch.object(analysis_utils, "analyze_text", return_value=fake_metrics):
        analyzed_df = analysis_utils.process_logs(fake_streamlit.session_state["all_logs"])
    enriched_df = analysis_utils.process_custom_lexicon(analyzed_df, {"Positive": ["mocked"]})

    assert len(enriched_df) == 2
    assert "token_count" in enriched_df.columns
    assert "Positive" in enriched_df.columns
