"""Simulation configuration model and environment loader."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

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


def load_json_config(file_path: str) -> Dict[str, Any]:
    """Load user-defined configuration from a JSON file.

    Args:
        file_path: Path to the JSON configuration file.

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
        with open(path, "r", encoding="utf-8-sig") as file:
            return json.load(file) or {}
    except (json.JSONDecodeError, OSError) as exception:
        logger.error(f"Failed to load configuration from '{file_path}': {exception}")
        return {}


REQUIRED_SECRETS = {
    "OPENROUTER_API_KEY": {
        "help": "Your OpenRouter API key. Get one at https://openrouter.ai/keys"
    },
}

OPTIONAL_SECRETS = {
    "SUPABASE_URL": {
        "help": "Supabase project URL (optional — cloud logging disabled if missing)"
    },
    "SUPABASE_ANON_KEY": {
        "help": "Supabase anonymous key (optional — cloud logging disabled if missing)"
    },
}


def validate_secrets() -> None:
    """Verify that required environment secrets are present and log availability.

    Missing required secrets cause an immediate failure with a clear message.
    Missing optional secrets only produce a warning so the simulation can still
    run in local or degraded mode.

    Raises:
        ValueError: If any required secret is missing or empty.
    """
    for name, meta in REQUIRED_SECRETS.items():
        value = os.getenv(name, "")
        if not value.strip():
            logger.critical(
                "missing_required_secret | secret=%s | hint=%s",
                name,
                meta["help"],
            )
            raise ValueError(f"Required secret '{name}' is not set. {meta['help']}")
        logger.info("secret_present | secret=%s", name)

    for name, meta in OPTIONAL_SECRETS.items():
        value = os.getenv(name, "")
        if not value.strip():
            logger.warning(
                "missing_optional_secret | secret=%s | hint=%s",
                name,
                meta["help"],
            )
        else:
            logger.info("secret_present | secret=%s", name)


@dataclass(frozen=True)
class SimulationConfig:
    """Holds configuration parameters for a complete simulation run."""

    # User defined parameters (loaded from JSON)
    model_a: str
    model_b: str
    persona_a: str
    persona_b: str
    num_turns: int
    initial_message: str
    batch_size: int
    max_tokens: int

    temperature_a: float
    temperature_b: float
    context_window: int

    # Secrets (loaded from environment)
    openrouter_api_key: str
    supabase_url: str
    supabase_anon_key: str

    @classmethod
    def load(cls, json_path: str = "config/simulation.json") -> "SimulationConfig":
        """Load the simulation configuration from JSON and environment variables.

        Args:
            json_path: The path to the user-defined JSON configuration file.

        Returns:
            A populated SimulationConfig instance.

        Raises:
            ValueError: If a required environment secret is missing or empty.
        """
        load_environment_variables()
        validate_secrets()
        config_data = load_json_config(json_path)

        # Extraction from JSON with defaults
        agents = config_data.get("agents", {})
        agent_a = agents.get("agent_a", {})
        agent_b = agents.get("agent_b", {})
        sim = config_data.get("simulation", {})

        return cls(
            model_a=agent_a.get("model", "google/gemma-3n-e4b-it"),
            model_b=agent_b.get("model", "google/gemma-3n-e4b-it"),
            persona_a=agent_a.get("persona", "Chief Technology Officer"),
            persona_b=agent_b.get("persona", "Financial Analyst"),
            temperature_a=float(agent_a.get("temperature", 1.0)),
            temperature_b=float(agent_b.get("temperature", 1.0)),
            num_turns=int(sim.get("num_turns", 10)),
            initial_message=sim.get("initial_message", "Hello."),
            batch_size=int(os.getenv("BATCH_SIZE", "10")),
            max_tokens=int(sim.get("max_tokens", 1000)),
            context_window=int(sim.get("context_window", 5)),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            supabase_url=os.getenv("SUPABASE_URL", ""),
            supabase_anon_key=os.getenv("SUPABASE_ANON_KEY", ""),
        )
