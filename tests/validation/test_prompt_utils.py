import pytest

from parrotlm.validation.prompt_utils import (
    compose_persona_from_traits,
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


def test_compose_persona_from_traits_combines_role_and_traits():
    persona = compose_persona_from_traits("CTO", ["skeptical", " curious "])
    assert "CTO" in persona
    assert "skeptical" in persona
    assert "curious" in persona


def test_compose_persona_from_traits_without_traits_returns_role():
    assert compose_persona_from_traits("CTO", []) == "CTO"


def test_compose_persona_from_traits_empty_role_defaults():
    persona = compose_persona_from_traits("  ", ["curious"])
    assert "conversation partner" in persona
