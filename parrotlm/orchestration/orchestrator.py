from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generator, Optional, Tuple

from parrotlm.infrastructure._logging import log_structured
from parrotlm.validation._validators import (
    normalize_response_data,
    validate_generation_parameters,
    validate_non_empty_string,
    validate_positive_int,
)
from parrotlm.agents.agent import Agent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentConfig:
    """Holds configuration parameters for initializing an agent."""

    model: str
    system_prompt: str
    user_persona_snapshot: str
    max_history_turns: int
    parameters: dict


class Orchestrator:
    """Manage one simulation run between two agents.

    This class handles the back-and-forth communication between two Agent instances,
    logging the conversation turns and handling stop conditions like agent refusals.
    """

    def __init__(
        self,
        agent_a_configuration: AgentConfig,
        agent_b_configuration: AgentConfig,
        scenario_name: str,
        openrouter_api_key: str,
        experiment_id: Optional[str] = None,
    ) -> None:
        """Initialize the Orchestrator with two agents and a scenario name.

        Args:
            agent_a_configuration: Configuration for the first agent.
            agent_b_configuration: Configuration for the second agent.
            scenario_name: Name of the scenario being run.
            openrouter_api_key: The API key for OpenRouter.
            experiment_id: Optional unique identifier for the experiment.
        """
        self.experiment_id = experiment_id or str(uuid.uuid4())
        self.scenario_name = validate_non_empty_string(scenario_name, "scenario_name")

        self.initialize_agent_instances(
            agent_a_configuration, agent_b_configuration, openrouter_api_key
        )
        self.validate_generation_parameters(
            agent_a_configuration, agent_b_configuration
        )

    def initialize_agent_instances(
        self,
        agent_a_configuration: AgentConfig,
        agent_b_configuration: AgentConfig,
        openrouter_api_key: str,
    ) -> None:
        """Create the Agent objects using their respective configurations."""
        max_history_turns_a = validate_positive_int(
            agent_a_configuration.max_history_turns,
            "agent_a_configuration.max_history_turns",
            default=20,
        )
        max_history_turns_b = validate_positive_int(
            agent_b_configuration.max_history_turns,
            "agent_b_configuration.max_history_turns",
            default=20,
        )

        self.agent_a = Agent(
            model_slug=agent_a_configuration.model,
            system_prompt=agent_a_configuration.system_prompt,
            name="Agent A",
            api_key=openrouter_api_key,
            max_history_turns=max_history_turns_a,
        )
        self.agent_b = Agent(
            model_slug=agent_b_configuration.model,
            system_prompt=agent_b_configuration.system_prompt,
            name="Agent B",
            api_key=openrouter_api_key,
            max_history_turns=max_history_turns_b,
        )

        self.persona_a_snapshot = (
            agent_a_configuration.user_persona_snapshot or self.agent_a.system_prompt
        )
        self.persona_b_snapshot = (
            agent_b_configuration.user_persona_snapshot or self.agent_b.system_prompt
        )

    def validate_generation_parameters(
        self, agent_a_configuration: AgentConfig, agent_b_configuration: AgentConfig
    ) -> None:
        """Validate and store the generation parameters for both agents."""
        self.agent_a_parameters = validate_generation_parameters(
            agent_a_configuration.parameters,
            "agent_a_configuration.parameters",
        )
        self.agent_b_parameters = validate_generation_parameters(
            agent_b_configuration.parameters,
            "agent_b_configuration.parameters",
        )

    def log_simulation_start(self, num_turns: int) -> None:
        """Record the start of the simulation in the structured logs."""
        log_structured(
            logging.INFO,
            "simulation_started",
            experiment_id=self.experiment_id,
            scenario=self.scenario_name,
            turns_requested=num_turns,
        )

    def log_simulation_completion(self, total_logs: int) -> None:
        """Record the successful completion of the simulation."""
        log_structured(
            logging.INFO,
            "simulation_completed",
            experiment_id=self.experiment_id,
            generated_log_entries=total_logs,
        )

    def run_simulation(
        self,
        num_turns: int,
        initial_message: str = "Hello.",
        cancellation_requested: Optional[Callable[[], bool]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Run a multi-turn conversation and yield log entries as they are created.

        Args:
            num_turns: The number of back-and-forth turns to execute.
            initial_message: The starting message for the conversation.

        Returns:
            A generator that yields structured log dictionaries for each agent's response.

        Raises:
            ValueError: If num_turns is not a positive integer.
        """
        if not isinstance(num_turns, int) or num_turns <= 0:
            raise ValueError("`num_turns` must be a positive integer.")
        last_message = validate_non_empty_string(initial_message, "initial_message")

        self.log_simulation_start(num_turns)

        total_logs = 0
        for log_entry in self.process_conversation_turns(
            num_turns, last_message, cancellation_requested
        ):
            yield log_entry
            total_logs += 1

        self.log_simulation_completion(total_logs)

    def process_conversation_turns(
        self,
        num_turns: int,
        initial_message: str,
        cancellation_requested: Optional[Callable[[], bool]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Iterate through the specified number of turns, managing handoffs between agents."""
        last_message = initial_message
        for turn_index in range(num_turns):
            if cancellation_requested and cancellation_requested():
                logger.info("Simulation cancelled before turn %s.", turn_index)
                break

            log_entry_a, last_message, should_stop = self._run_single_agent_turn(
                turn_index=turn_index,
                speaker=self.agent_a,
                responder=self.agent_b,
                system_prompt_snapshot=self.persona_a_snapshot,
                input_message=last_message,
                generation_parameters=self.agent_a_parameters,
            )
            yield log_entry_a
            if should_stop or (cancellation_requested and cancellation_requested()):
                break

            log_entry_b, last_message, should_stop = self._run_single_agent_turn(
                turn_index=turn_index,
                speaker=self.agent_b,
                responder=self.agent_a,
                system_prompt_snapshot=self.persona_b_snapshot,
                input_message=last_message,
                generation_parameters=self.agent_b_parameters,
            )
            yield log_entry_b
            if should_stop:
                break

    def request_agent_generation(
        self,
        speaker: Agent,
        input_message: str,
        generation_parameters: Dict[str, Any],
        turn_index: int,
    ) -> Any:
        """Ask the speaker agent to generate a response based on the input."""
        try:
            logger.info("Turn %s: %s generating response...", turn_index, speaker.name)
            return speaker.generate_response(input_message, **generation_parameters)
        except Exception as exception:
            logger.exception("Failed turn %s for %s.", turn_index, speaker.name)
            raise RuntimeError(
                f"{speaker.name} failed on turn {turn_index}."
            ) from exception

    def normalize_agent_payload(
        self, speaker: Agent, response_data: Any, turn_index: int
    ) -> Dict[str, Any]:
        """Validate and normalize the raw response data from the agent."""
        try:
            return normalize_response_data(response_data)
        except (KeyError, TypeError, ValueError) as exception:
            logger.exception(
                "invalid_response_payload | speaker=%s turn_index=%s",
                speaker.name,
                turn_index,
            )
            raise RuntimeError(
                f"{speaker.name} returned an invalid payload on turn {turn_index}."
            ) from exception

    def evaluate_stop_condition(
        self, turn_index: int, speaker_name: str, is_refusal: bool
    ) -> bool:
        """Check if the conversation should stop due to a model refusal."""
        should_stop = bool(is_refusal)
        if should_stop:
            log_structured(
                logging.WARNING,
                "agent_refusal_detected",
                experiment_id=self.experiment_id,
                turn_index=turn_index,
                speaker=speaker_name,
            )
        return should_stop

    def _run_single_agent_turn(
        self,
        turn_index: int,
        speaker: Agent,
        responder: Agent,
        system_prompt_snapshot: str,
        input_message: str,
        generation_parameters: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str, bool]:
        """Generate one speaker response, append the log, and return stop status."""
        response_data = self.request_agent_generation(
            speaker, input_message, generation_parameters, turn_index
        )
        normalized_response_data = self.normalize_agent_payload(
            speaker, response_data, turn_index
        )

        log_entry = self._create_log_entry(
            turn_index=turn_index,
            speaker=speaker,
            responder=responder,
            response_data=normalized_response_data,
            system_prompt_snapshot=system_prompt_snapshot,
            input_message=input_message,
        )

        # We must keep a non-empty handoff message so the next agent receives valid input.

        # Sending a completely blank message to the next agent breaks the strict role alternation
        # sequence that OpenRouter expects, causing subsequent API requests to fail.
        next_message = normalized_response_data["content"] or "..."
        should_stop = self.evaluate_stop_condition(
            turn_index, speaker.name, normalized_response_data["is_refusal"]
        )

        return log_entry, next_message, should_stop

    def format_turn_metrics(
        self, response_data: Dict[str, Any], input_message: str
    ) -> Dict[str, Any]:
        """Format the specific metrics from the response data into the log schema."""
        return {
            "latency_ms": response_data["latency_ms"],
            "input_tokens": response_data["input_tokens"],
            "output_tokens": response_data["output_tokens"],
            "input_preview": input_message[:120],
            "content": response_data["content"],
            "finish_reason": response_data["finish_reason"],
            "is_refusal": response_data["is_refusal"],
        }

    def assemble_log_record(
        self,
        turn_index: int,
        speaker: Agent,
        responder: Agent,
        system_prompt_snapshot: str,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Combine all context into a single log record dictionary."""
        speaker_slot = "A" if speaker is self.agent_a else "B"
        record = {
            "experiment_id": self.experiment_id,
            "turn_id": turn_index,  # keep standard field name for database
            "scenario": self.scenario_name,
            "speaker_slot": speaker_slot,
            "speaker_name": speaker.name,
            "speaker_model": speaker.model_slug,
            "responder_model": responder.model_slug,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_prompt_snapshot": system_prompt_snapshot,
        }
        record.update(metrics)
        return record

    def _create_log_entry(
        self,
        turn_index: int,
        speaker: Agent,
        responder: Agent,
        response_data: Dict[str, Any],
        system_prompt_snapshot: str,
        input_message: str,
    ) -> Dict[str, Any]:
        """Create a normalized log dictionary for one model output."""
        metrics = self.format_turn_metrics(response_data, input_message)
        return self.assemble_log_record(
            turn_index, speaker, responder, system_prompt_snapshot, metrics
        )
