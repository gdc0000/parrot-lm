import pytest

from parrotlm.prompt_utils import (
    construct_system_prompt,
    retrieve_dialogue_formatting_rules,
    format_persona_instructions,
)


def test_construct_system_prompt_includes_persona_and_rules():
    persona = "A curious engineer"
    prompt = construct_system_prompt(persona)

    assert persona in prompt
    assert "DIALOGUE ONLY" in prompt
    assert "YOUR PERSONA" in prompt


def test_construct_system_prompt_rejects_empty_persona():
    with pytest.raises(ValueError, match="must be a non-empty string"):
        construct_system_prompt("   ")


def test_construct_system_prompt_rejects_non_string_persona():
    with pytest.raises(ValueError, match="must be a non-empty string"):
        construct_system_prompt(123)  # type: ignore[arg-type]


def test_retrieve_dialogue_formatting_rules():
    rules = retrieve_dialogue_formatting_rules()
    assert "NO asterisks" in rules


def test_format_persona_instructions():
    result = format_persona_instructions("rules", "persona")
    assert "rules" in result
    assert "persona" in result
