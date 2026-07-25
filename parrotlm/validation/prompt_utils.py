"""Prompt construction helpers."""


def retrieve_dialogue_formatting_rules() -> str:
    """Retrieve the static formatting rules for dialogue generation.

    Returns:
        The formatting rules as a string.
    """
    # We explicitly forbid asterisks and brackets because models tend to use them
    # to act as narrators or provide stage directions (e.g., *smiles*), breaking the immersion.
    dialogue_formatting_rules = """
# MANDATORY: DIALOGUE ONLY. ZERO NARRATION.
You are a human. You are NOT writing a script. You are NOT a narrator.

## FORBIDDEN SYMBOLS:
- NO parentheses: ( )
- NO asterisks: * *
- NO brackets: [ ]
- NO formatting markers for actions.

## FORBIDDEN BEHAVIORS:
- NEVER describe actions (for example, "I smile", "leans in").
- NEVER describe feelings as stage directions (for example, "sighs").
- NEVER use non-spoken scene text.

## ONLY WHAT IS SPOKEN:
Your response must contain only the words spoken by the character.
If an action occurs, imply it through spoken language only.

## CONVERSATIONAL STYLE:
- Use natural human dialogue flow.
- You may use filler words and brief hesitations.
- Keep the character voice consistent.
- Respond to what the other person said; do not copy their last message verbatim.
- Add at least one new idea, reaction, or question in each reply.
""".strip()
    return dialogue_formatting_rules


def format_persona_instructions(dialogue_formatting_rules: str, persona: str) -> str:
    """Format the complete system prompt using rules and persona.

    Args:
        dialogue_formatting_rules: The dialogue constraints to enforce.
        persona: The specific persona to adopt.

    Returns:
        The complete formatted system prompt string.
    """
    return (
        f"{dialogue_formatting_rules}\n\n"
        f"YOUR PERSONA:\n{persona.strip()}\n\n"
        "FINAL WARNING: Output only spoken words from the character."
    )


def compose_persona_from_traits(role: str, traits: list) -> str:
    """Compose a persona string from a role and selected psychological traits.

    Args:
        role: The character's role or occupation (e.g. 'Chief Technology Officer').
        traits: A list of psychological trait descriptors (e.g. ['skeptical', 'curious']).

    Returns:
        A persona sentence combining role and traits, suitable for `construct_system_prompt`.
    """
    clean_role = (role or "").strip() or "conversation partner"
    clean_traits = [trait.strip() for trait in traits if isinstance(trait, str) and trait.strip()]
    if not clean_traits:
        return clean_role
    return (
        f"{clean_role}. Personality traits: {', '.join(clean_traits)}. "
        "Let these traits shape your tone, opinions, and reactions."
    )


def construct_system_prompt(persona: str) -> str:
    """Build a dialogue-only system prompt with the provided persona context.

    Args:
        persona: The specific persona the agent should adopt during the simulation.

    Returns:
        The fully formatted system prompt combining dialogue rules and persona.

    Raises:
        ValueError: If `persona` is not a string or if the stripped string is empty.
    """
    if not isinstance(persona, str) or not persona.strip():
        raise ValueError(
            f"`persona` must be a non-empty string. Received type: {type(persona).__name__}, value: {persona}"
        )

    dialogue_formatting_rules = retrieve_dialogue_formatting_rules()
    return format_persona_instructions(dialogue_formatting_rules, persona)
