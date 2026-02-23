from parrotlm.simulation_config import SimulationConfig


def test_from_env_returns_simulation_config_instance():
    config = SimulationConfig.from_env()
    assert isinstance(config, SimulationConfig)


def test_from_env_uses_non_empty_string_fields_when_env_is_set(monkeypatch):
    monkeypatch.setenv("MODEL_A", "openai/gpt-4o-mini")
    monkeypatch.setenv("MODEL_B", "openai/gpt-4o-mini")
    monkeypatch.setenv("PERSONA_A", "Chief Technology Officer")
    monkeypatch.setenv("PERSONA_B", "Financial Analyst")
    monkeypatch.setenv("INITIAL_MESSAGE", "Start the discussion.")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "example-key")

    config = SimulationConfig.from_env()

    assert config.model_a
    assert config.model_b
    assert config.persona_a
    assert config.persona_b
    assert config.initial_message
    assert config.supabase_url
    assert config.supabase_key

