import yaml
from parrotlm.simulation_config import SimulationConfig


def test_load_returns_simulation_config_instance():
    config = SimulationConfig.load()
    assert isinstance(config, SimulationConfig)


def test_load_from_yaml(tmp_path):
    config_file = tmp_path / "test_config.yaml"
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
    with open(config_file, "w") as f:
        yaml.dump(data, f)

    config = SimulationConfig.load(yaml_path=str(config_file))

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
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-key")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")

    config_file = tmp_path / "empty.yaml"
    with open(config_file, "w") as f:
        f.write("{}")

    config = SimulationConfig.load(yaml_path=str(config_file))
    assert config.openrouter_api_key == "secret-key"
    assert config.supabase_url == "https://example.supabase.co"
    assert config.supabase_anon_key == "anon-key"


def test_load_with_missing_yaml(caplog):
    # Should use defaults
    config = SimulationConfig.load(yaml_path="non_existent.yaml")
    assert config.model_a == "openai/gpt-4o-mini"
    assert "Configuration file 'non_existent.yaml' not found" in caplog.text
