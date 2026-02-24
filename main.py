from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from parrotlm._logging import log_structured, setup_logging
from parrotlm.orchestrator import AgentConfig, Orchestrator
from parrotlm.prompt_utils import construct_system_prompt
from parrotlm.simulation_config import SimulationConfig
from parrotlm.supabase_client import get_supabase_client
from parrotlm.supabase_logger import upload_session_logs


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
) -> List[Dict[str, Any]]:
    """Run the complete agent interaction scenario and collect the logs.

    Args:
        agent_a_configuration: Setup for the first agent.
        agent_b_configuration: Setup for the second agent.
        configuration: Global settings containing API keys and turn limits.

    Returns:
        A list of generated log entries from the simulation.
    """
    orchestrator = Orchestrator(
        agent_a_configuration=agent_a_configuration,
        agent_b_configuration=agent_b_configuration,
        scenario_name="simulation",
        openrouter_api_key=configuration.openrouter_api_key,
    )

    # We wrap the generator in list() to force the entire simulation to evaluate
    # synchronously. We must collect all log entries before attempting the batch
    # upload to Supabase in the next step.
    return list(
        orchestrator.run_simulation(
            num_turns=configuration.num_turns,
            initial_message=configuration.initial_message,
        )
    )


def process_simulation_results(logs: List[Dict[str, Any]]) -> None:
    """Handle post-simulation tasks like uploading logs and emitting final metrics.

    Args:
        logs: The collection of interaction records from the simulation.
    """
    upload_session_logs(logs)
    log_structured(logging.INFO, "simulation_complete", num_logs=len(logs))


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
        logs = execute_simulation(agent_a, agent_b, configuration)

        current_phase = "result_processing"
        process_simulation_results(logs)

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
