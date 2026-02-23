"""Simulation configuration model and environment loader."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class SimulationConfig:
    model_a: str
    model_b: str
    persona_a: str
    persona_b: str
    num_turns: int
    initial_message: str
    max_tokens: int
    temperature_a: float
    temperature_b: float
    context_window: int
    supabase_url: str
    supabase_key: str

    @classmethod
    def from_env(cls) -> "SimulationConfig":
        return cls(
            model_a=os.getenv("MODEL_A", "openai/gpt-4o-mini"),
            model_b=os.getenv("MODEL_B", "openai/gpt-4o-mini"),
            persona_a=os.getenv("PERSONA_A", "Chief Technology Officer"),
            persona_b=os.getenv("PERSONA_B", "Financial Analyst"),
            num_turns=_get_int("NUM_TURNS", 10),
            initial_message=os.getenv(
                "INITIAL_MESSAGE",
                "What is your outlook on AI investment over the next 12 months?",
            ),
            max_tokens=_get_int("MAX_TOKENS", 1000),
            temperature_a=_get_float("TEMPERATURE_A", 1.0),
            temperature_b=_get_float("TEMPERATURE_B", 1.0),
            context_window=_get_int("CONTEXT_WINDOW", 5),
            supabase_url=os.getenv("SUPABASE_URL", ""),
            supabase_key=os.getenv("SUPABASE_KEY", ""),
        )
