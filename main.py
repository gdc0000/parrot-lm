from __future__ import annotations

import logging

from parrotlm._logging import log_structured, setup_logging
from parrotlm.orchestrator import AgentConfig, Orchestrator
from parrotlm.simulation_config import SimulationConfig
from parrotlm.supabase_logger import upload_session_logs


def main() -> None:
    try:
        setup_logging()

        config = SimulationConfig.from_env()

        agent_a_config = AgentConfig(
            model=config.model_a,
            system_prompt=config.persona_a,
            user_persona_snapshot=config.persona_a,
            max_history_turns=config.context_window,
            params={
                "max_tokens": config.max_tokens,
                "temperature": config.temperature_a,
            },
        )
        agent_b_config = AgentConfig(
            model=config.model_b,
            system_prompt=config.persona_b,
            user_persona_snapshot=config.persona_b,
            max_history_turns=config.context_window,
            params={
                "max_tokens": config.max_tokens,
                "temperature": config.temperature_b,
            },
        )

        orchestrator = Orchestrator(
            agent_a_config=agent_a_config,
            agent_b_config=agent_b_config,
            scenario_name="simulation",
        )

        logs = list(
            orchestrator.run_simulation(
                num_turns=config.num_turns,
                initial_message=config.initial_message,
            )
        )

        upload_session_logs(logs)

        log_structured(logging.INFO, "simulation_complete", num_logs=len(logs))
    except Exception as exc:
        log_structured(
            logging.CRITICAL,
            "unhandled_exception",
            error=str(exc),
            exception_type=type(exc).__name__,
        )
        raise


if __name__ == "__main__":
    main()
