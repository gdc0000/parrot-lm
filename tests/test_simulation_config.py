from parrotlm.simulation_config import SimulationConfig, _get_int, _get_float


def test_from_env_returns_simulation_config_instance():
    config = SimulationConfig.from_env()
    assert isinstance(config, SimulationConfig)


def test_from_env_uses_non_empty_string_fields_when_env_is_set(monkeypatch):
    monkeypatch.setenv("MODEL_A", "openai/gpt-4o-mini")
    monkeypatch.setenv("MODEL_B", "openai/gpt-4o-mini")
    monkeypatch.setenv("PERSONA_A", "Chief Technology Officer")
    monkeypatch.setenv("PERSONA_B", "Financial Analyst")
    monkeypatch.setenv("INITIAL_MESSAGE", "Start the discussion.")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-api-key")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "example-key")

    config = SimulationConfig.from_env()

    assert config.model_a
    assert config.model_b
    assert config.persona_a
    assert config.persona_b
    assert config.initial_message
    assert config.openrouter_api_key == "test-api-key"
    assert config.supabase_url
    assert config.supabase_anon_key == "example-key"


def test_get_int_failure(monkeypatch, caplog):
    monkeypatch.setenv("TEST_INT", "not-a-number")

    with caplog.at_level("WARNING"):
        result = _get_int("TEST_INT", 42)

    assert result == 42
    assert (
        "Failed to parse environment variable 'TEST_INT' as an integer" in caplog.text
    )


def test_get_float_failure(monkeypatch, caplog):
    monkeypatch.setenv("TEST_FLOAT", "not-a-float")

    with caplog.at_level("WARNING"):
        result = _get_float("TEST_FLOAT", 3.14)

    assert result == 3.14
    assert "Failed to parse environment variable 'TEST_FLOAT' as a float" in caplog.text
