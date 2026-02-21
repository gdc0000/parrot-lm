import pytest

from parrotlm.prompt_utils import construct_system_prompt


def test_construct_system_prompt_includes_persona_and_rules():
    persona = "A curious engineer"
    prompt = construct_system_prompt(persona)

    assert persona in prompt
    assert "DIALOGUE ONLY" in prompt
    assert "YOUR PERSONA" in prompt


def test_construct_system_prompt_rejects_empty_persona():
    with pytest.raises(ValueError):
        construct_system_prompt("   ")


def test_construct_system_prompt_rejects_non_string_persona():
    with pytest.raises(ValueError):
        construct_system_prompt(123)  # type: ignore[arg-type]

