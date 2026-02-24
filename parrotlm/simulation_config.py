"""Simulation configuration model and environment loader."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _get_int(name: str, default: int) -> int:
    """Safely fetch and parse an integer from environment variables.

    Args:
        name: The name of the environment variable to fetch.
        default: The fallback value if the variable is missing or invalid.

    Returns:
        The parsed integer or the default value.
    """
    raw_environment_value = os.getenv(name)
    if raw_environment_value is None or not raw_environment_value.strip():
        return default
    try:
        return int(raw_environment_value)
    except ValueError:
        logger.warning(
            "Failed to parse environment variable '%s' as an integer. Received: '%s'. "
            "Falling back to default: %s.",
            name,
            raw_environment_value,
            default,
        )
        return default


def _get_float(name: str, default: float) -> float:
    """Safely fetch and parse a float from environment variables.

    Args:
        name: The name of the environment variable to fetch.
        default: The fallback value if the variable is missing or invalid.

    Returns:
        The parsed float or the default value.
    """
    raw_environment_value = os.getenv(name)
    if raw_environment_value is None or not raw_environment_value.strip():
        return default
    try:
        return float(raw_environment_value)
    except ValueError:
        logger.warning(
            "Failed to parse environment variable '%s' as a float. Received: '%s'. "
            "Falling back to default: %s.",
            name,
            raw_environment_value,
            default,
        )
        return default


def load_environment_variables() -> None:
    """Attempt to load environment variables from a .env file if available.

    We use a try-except block here so that local development can use the
    python-dotenv library for convenience, while production environments
    (where environment variables are injected directly) do not crash if
    the library is omitted from dependencies.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        logger.debug(
            "python-dotenv library is not installed. Skipping .env file loading."
        )


def instantiate_configuration(
    class_reference: type[SimulationConfig],
) -> SimulationConfig:
    """Instantiate the SimulationConfig object using current environment variables.

    Args:
        class_reference: The class to instantiate (typically SimulationConfig).

    Returns:
        A fully populated SimulationConfig instance.
    """
    return class_reference(
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
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        supabase_url=os.getenv("SUPABASE_URL", ""),
        supabase_anon_key=os.getenv("SUPABASE_ANON_KEY", ""),
    )


@dataclass(frozen=True)
class SimulationConfig:
    """Holds configuration parameters for a complete simulation run."""

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
    openrouter_api_key: str
    supabase_url: str
    supabase_anon_key: str

    @classmethod
    def from_env(cls) -> "SimulationConfig":
        """Load the simulation configuration from environment variables.

        Returns:
            A populated SimulationConfig instance.
        """
        load_environment_variables()
        return instantiate_configuration(cls)
