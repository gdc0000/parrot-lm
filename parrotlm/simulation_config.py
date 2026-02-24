"""Simulation configuration model and environment loader."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


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


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Load user-defined configuration from a YAML file.

    Args:
        file_path: Path to the YAML configuration file.

    Returns:
        A dictionary containing the configuration data, or an empty dict if not found.
    """
    path = Path(file_path)
    if not path.exists():
        logger.warning(
            f"Configuration file '{file_path}' not found. Using internal defaults."
        )
        return {}

    try:
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    except (yaml.YAMLError, OSError) as exception:
        logger.error(f"Failed to load configuration from '{file_path}': {exception}")
        return {}


@dataclass(frozen=True)
class SimulationConfig:
    """Holds configuration parameters for a complete simulation run."""

    # User defined parameters (loaded from YAML)
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

    # Secrets (loaded from environment)
    openrouter_api_key: str
    supabase_url: str
    supabase_anon_key: str

    @classmethod
    def load(cls, yaml_path: str = "config/simulation.yaml") -> "SimulationConfig":
        """Load the simulation configuration from YAML and environment variables.

        Args:
            yaml_path: The path to the user-defined YAML configuration file.

        Returns:
            A populated SimulationConfig instance.
        """
        load_environment_variables()
        config_data = load_yaml_config(yaml_path)

        # Extraction from YAML with defaults
        agents = config_data.get("agents", {})
        agent_a = agents.get("agent_a", {})
        agent_b = agents.get("agent_b", {})
        sim = config_data.get("simulation", {})

        return cls(
            model_a=agent_a.get("model", "openai/gpt-4o-mini"),
            model_b=agent_b.get("model", "openai/gpt-4o-mini"),
            persona_a=agent_a.get("persona", "Chief Technology Officer"),
            persona_b=agent_b.get("persona", "Financial Analyst"),
            temperature_a=float(agent_a.get("temperature", 1.0)),
            temperature_b=float(agent_b.get("temperature", 1.0)),
            num_turns=int(sim.get("num_turns", 10)),
            initial_message=sim.get("initial_message", "Hello."),
            max_tokens=int(sim.get("max_tokens", 1000)),
            context_window=int(sim.get("context_window", 5)),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            supabase_url=os.getenv("SUPABASE_URL", ""),
            supabase_anon_key=os.getenv("SUPABASE_ANON_KEY", ""),
        )
