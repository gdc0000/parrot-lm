"""Prompt construction helpers."""


def construct_system_prompt(persona: str) -> str:
    """Build a dialogue-only system prompt with the provided persona context."""
    if not isinstance(persona, str) or not persona.strip():
        raise ValueError("`persona` must be a non-empty string.")

    dialogue_rules = """
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
""".strip()

    return (
        f"{dialogue_rules}\n\n"
        f"YOUR PERSONA:\n{persona.strip()}\n\n"
        "FINAL WARNING: Output only spoken words from the character."
    )
