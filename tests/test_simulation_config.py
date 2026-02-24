import json
import pytest
from parrotlm.simulation_config import SimulationConfig


def test_load_returns_simulation_config_instance():
    config = SimulationConfig.load()
    assert isinstance(config, SimulationConfig)


def test_load_from_json(tmp_path):
    config_file = tmp_path / "test_config.json"
    data = {
        "agents": {
            "agent_a": {"model": "model-a", "persona": "persona-a", "temperature": 0.5},
            "agent_b": {"model": "model-b", "persona": "persona-b", "temperature": 0.8},
        },
        "simulation": {
            "num_turns": 5,
            "initial_message": "test message",
            "max_tokens": 100,
            "context_window": 2,
        },
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    config = SimulationConfig.load(json_path=str(config_file))

    assert config.model_a == "model-a"
    assert config.persona_a == "persona-a"
    assert config.temperature_a == 0.5
    assert config.model_b == "model-b"
    assert config.persona_b == "persona-b"
    assert config.temperature_b == 0.8
    assert config.num_turns == 5
    assert config.initial_message == "test message"
    assert config.max_tokens == 100
    assert config.context_window == 2


def test_load_uses_secrets_from_env(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-key")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")

    config_file = tmp_path / "empty.json"
    with open(config_file, "w", encoding="utf-8") as f:
        f.write("{}")

    config = SimulationConfig.load(json_path=str(config_file))
    assert config.openrouter_api_key == "secret-key"
    assert config.supabase_url == "https://example.supabase.co"
    assert config.supabase_anon_key == "anon-key"


def test_load_with_missing_json(caplog, monkeypatch):
    monkeypatch.setattr("parrotlm.simulation_config.load_environment_variables", lambda: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)

    # Should use defaults
    config = SimulationConfig.load(json_path="non_existent.json")
    assert config.model_a == "google/gemma-3n-e4b-it"
    assert config.model_b == "google/gemma-3n-e4b-it"
    assert config.context_window == 5
    assert config.openrouter_api_key == ""
    assert config.supabase_url == ""
    assert config.supabase_anon_key == ""
    assert "Configuration file 'non_existent.json' not found" in caplog.text


def test_load_with_missing_env_returns_empty_secrets(monkeypatch, tmp_path):
    monkeypatch.setattr("parrotlm.simulation_config.load_environment_variables", lambda: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)

    config_file = tmp_path / "empty.json"
    config_file.write_text("{}", encoding="utf-8")

    config = SimulationConfig.load(json_path=str(config_file))

    assert config.openrouter_api_key == ""
    assert config.supabase_url == ""
    assert config.supabase_anon_key == ""


def test_load_with_malformed_json_returns_defaults(tmp_path, caplog):
    config_file = tmp_path / "malformed.json"
    config_file.write_text("{ this is not valid json", encoding="utf-8")

    config = SimulationConfig.load(json_path=str(config_file))

    assert config.model_a == "google/gemma-3n-e4b-it"
    assert config.model_b == "google/gemma-3n-e4b-it"
    assert config.num_turns == 10
    assert "Failed to load configuration" in caplog.text


def test_load_with_partial_json_uses_defaults_for_missing_sections(tmp_path):
    config_file = tmp_path / "partial.json"
    data = {
        "agents": {"agent_a": {"model": "model-only-a"}},
        "simulation": {"context_window": 9},
    }
    config_file.write_text(json.dumps(data), encoding="utf-8")

    config = SimulationConfig.load(json_path=str(config_file))

    assert config.model_a == "model-only-a"
    assert config.model_b == "google/gemma-3n-e4b-it"
    assert config.persona_a == "Chief Technology Officer"
    assert config.persona_b == "Financial Analyst"
    assert config.temperature_a == 1.0
    assert config.temperature_b == 1.0
    assert config.context_window == 9


def test_load_raises_for_invalid_temperature_type(tmp_path):
    config_file = tmp_path / "invalid_types.json"
    data = {"agents": {"agent_a": {"temperature": "alta"}}}
    config_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError):
        SimulationConfig.load(json_path=str(config_file))
