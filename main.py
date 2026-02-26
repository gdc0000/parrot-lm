from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from parrotlm.infrastructure._logging import log_structured, setup_logging
from parrotlm.orchestration.orchestrator import AgentConfig, Orchestrator
from parrotlm.validation.prompt_utils import construct_system_prompt
from parrotlm.configuration.simulation_config import SimulationConfig
from parrotlm.infrastructure.supabase_client import get_supabase_client
from parrotlm.infrastructure.supabase_logger import SupabaseBufferedLogger


def initialize_infrastructure() -> SimulationConfig:
    """Set up logging and external database clients, and load configuration.

    Returns:
        The fully loaded SimulationConfig object.
    """
    setup_logging()
    configuration = SimulationConfig.load()
    get_supabase_client(
        url=configuration.supabase_url, key=configuration.supabase_anon_key
    )
    return configuration


def configure_simulation_agents(
    configuration: SimulationConfig,
) -> Tuple[AgentConfig, AgentConfig]:
    """Create the specific agent configurations for the simulation.

    Args:
        configuration: The global simulation settings.

    Returns:
        A tuple containing the AgentConfig for agent A and agent B.
    """
    agent_a_configuration = AgentConfig(
        model=configuration.model_a,
        system_prompt=construct_system_prompt(configuration.persona_a),
        user_persona_snapshot=configuration.persona_a,
        max_history_turns=configuration.context_window,
        parameters={
            "max_tokens": configuration.max_tokens,
            "temperature": configuration.temperature_a,
        },
    )
    agent_b_configuration = AgentConfig(
        model=configuration.model_b,
        system_prompt=construct_system_prompt(configuration.persona_b),
        user_persona_snapshot=configuration.persona_b,
        max_history_turns=configuration.context_window,
        parameters={
            "max_tokens": configuration.max_tokens,
            "temperature": configuration.temperature_b,
        },
    )
    return agent_a_configuration, agent_b_configuration


def execute_simulation(
    agent_a_configuration: AgentConfig,
    agent_b_configuration: AgentConfig,
    configuration: SimulationConfig,
) -> None:
    """Run the complete agent interaction scenario and stream logs to Supabase.

    Args:
        agent_a_configuration: Setup for the first agent.
        agent_b_configuration: Setup for the second agent.
        configuration: Global settings containing API keys and turn limits.
    """
    orchestrator = Orchestrator(
        agent_a_configuration=agent_a_configuration,
        agent_b_configuration=agent_b_configuration,
        scenario_name="simulation",
        openrouter_api_key=configuration.openrouter_api_key,
    )

    buffered_logger = SupabaseBufferedLogger(batch_size=configuration.batch_size)

    try:
        # We iterate over the generator to process logs one by one as they are produced.
        # This prevents memory exhaustion during long simulations.
        for log_entry in orchestrator.run_simulation(
            num_turns=configuration.num_turns,
            initial_message=configuration.initial_message,
        ):
            buffered_logger.push(log_entry)
            log_structured(
                logging.INFO, "log_entry_generated", turn_id=log_entry["turn_id"]
            )

    finally:
        # Ensure any remaining logs in the buffer are uploaded even if the simulation stops early.
        buffered_logger.flush()


def main() -> None:
    """Execute the main simulation pipeline from start to finish.

    This function sets up the infrastructure, configures the agents, runs
    the conversation loop, and processes the output.

    Raises:
        Exception: If any critical failure occurs during initialization,
            simulation, or processing. The error is logged securely before reraising.
    """
    current_phase = "initialization"
    try:
        configuration = initialize_infrastructure()

        current_phase = "agent_configuration"
        agent_a, agent_b = configure_simulation_agents(configuration)

        current_phase = "simulation_execution"
        execute_simulation(agent_a, agent_b, configuration)

    except Exception as exception:
        log_structured(
            logging.CRITICAL,
            "unhandled_exception",
            failed_phase=current_phase,
            error=str(exception),
            exception_type=type(exception).__name__,
        )
        raise


if __name__ == "__main__":
    main()
