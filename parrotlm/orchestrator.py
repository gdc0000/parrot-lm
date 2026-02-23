"""Orchestrator: coordinates a multi-turn conversation between two agents."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple

from parrotlm._logging import log_structured
from parrotlm._validators import (
    normalize_response_data,
    validate_generation_params,
    validate_non_empty_string,
    validate_positive_int,
)
from parrotlm.agent import Agent

# Re-export Agent so that `from parrotlm.orchestrator import Agent` still works.
__all__ = ["Agent", "Orchestrator"]

logger = logging.getLogger(__name__)


class Orchestrator:
    """Manage one simulation run between two agents."""

    def __init__(
        self,
        agent_a_config: Dict[str, Any],
        agent_b_config: Dict[str, Any],
        scenario_name: str,
        experiment_id: Optional[str] = None,
    ) -> None:
        self.experiment_id = experiment_id or str(uuid.uuid4())
        self.scenario_name = validate_non_empty_string(scenario_name, "scenario_name")

        if not isinstance(agent_a_config, dict) or not isinstance(agent_b_config, dict):
            raise TypeError("`agent_a_config` and `agent_b_config` must be dictionaries.")

        max_history_turns_a = validate_positive_int(
            agent_a_config.get("max_history_turns"),
            "agent_a_config['max_history_turns']",
            default=20,
        )
        max_history_turns_b = validate_positive_int(
            agent_b_config.get("max_history_turns"),
            "agent_b_config['max_history_turns']",
            default=20,
        )

        self.agent_a = Agent(
            model_slug=agent_a_config["model"],
            system_prompt=agent_a_config["system_prompt"],
            name="Agent A",
            max_history_turns=max_history_turns_a,
        )
        self.agent_b = Agent(
            model_slug=agent_b_config["model"],
            system_prompt=agent_b_config["system_prompt"],
            name="Agent B",
            max_history_turns=max_history_turns_b,
        )

        self.persona_a_snapshot = agent_a_config.get("user_persona_snapshot", self.agent_a.system_prompt)
        self.persona_b_snapshot = agent_b_config.get("user_persona_snapshot", self.agent_b.system_prompt)
        self.agent_a_params = validate_generation_params(
            agent_a_config.get("params"),
            "agent_a_config['params']",
        )
        self.agent_b_params = validate_generation_params(
            agent_b_config.get("params"),
            "agent_b_config['params']",
        )
        self.logs: List[Dict[str, Any]] = []

    def run_simulation(
        self,
        num_turns: int,
        initial_message: str = "Hello.",
    ) -> Generator[Dict[str, Any], None, None]:
        """Run a multi-turn conversation and yield log entries as they are created."""
        if not isinstance(num_turns, int) or num_turns <= 0:
            raise ValueError("`num_turns` must be a positive integer.")
        last_message = validate_non_empty_string(initial_message, "initial_message")

        log_structured(
            logging.INFO,
            "simulation_started",
            experiment_id=self.experiment_id,
            scenario=self.scenario_name,
            turns_requested=num_turns,
        )

        for turn_id in range(num_turns):
            log_entry_a, last_message, should_stop = self._run_single_agent_turn(
                turn_id=turn_id,
                speaker=self.agent_a,
                responder=self.agent_b,
                system_prompt_snapshot=self.persona_a_snapshot,
                input_message=last_message,
                generation_params=self.agent_a_params,
            )
            yield log_entry_a
            if should_stop:
                break

            log_entry_b, last_message, should_stop = self._run_single_agent_turn(
                turn_id=turn_id,
                speaker=self.agent_b,
                responder=self.agent_a,
                system_prompt_snapshot=self.persona_b_snapshot,
                input_message=last_message,
                generation_params=self.agent_b_params,
            )
            yield log_entry_b
            if should_stop:
                break

        log_structured(
            logging.INFO,
            "simulation_completed",
            experiment_id=self.experiment_id,
            generated_log_entries=len(self.logs),
        )

    def _run_single_agent_turn(
        self,
        turn_id: int,
        speaker: Agent,
        responder: Agent,
        system_prompt_snapshot: str,
        input_message: str,
        generation_params: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str, bool]:
        """Generate one speaker response, append the log, and return stop status."""
        try:
            logger.info("Turn %s: %s generating response...", turn_id, speaker.name)
            response_data = speaker.generate_response(input_message, **generation_params)
        except Exception as exception:
            logger.exception("Failed turn %s for %s.", turn_id, speaker.name)
            raise RuntimeError(f"{speaker.name} failed on turn {turn_id}.") from exception

        try:
            normalized_response_data = normalize_response_data(response_data)
        except (KeyError, TypeError, ValueError) as exception:
            logger.exception(
                "invalid_response_payload | speaker=%s turn_id=%s",
                speaker.name,
                turn_id,
            )
            raise RuntimeError(f"{speaker.name} returned an invalid payload on turn {turn_id}.") from exception

        log_entry = self._create_log_entry(
            turn_id=turn_id,
            speaker=speaker,
            responder=responder,
            response_data=normalized_response_data,
            system_prompt_snapshot=system_prompt_snapshot,
            input_message=input_message,
        )
        self.logs.append(log_entry)

        # Keep a non-empty handoff message so the next agent receives valid input even on blank output.
        next_message = normalized_response_data["content"] or "..."
        should_stop = bool(normalized_response_data["is_refusal"])
        if should_stop:
            log_structured(
                logging.WARNING,
                "agent_refusal_detected",
                experiment_id=self.experiment_id,
                turn_id=turn_id,
                speaker=speaker.name,
            )

        return log_entry, next_message, should_stop

    def _create_log_entry(
        self,
        turn_id: int,
        speaker: Agent,
        responder: Agent,
        response_data: Dict[str, Any],
        system_prompt_snapshot: str,
        input_message: str,
    ) -> Dict[str, Any]:
        """Create a normalized log dictionary for one model output."""
        speaker_slot = "A" if speaker is self.agent_a else "B"
        return {
            "experiment_id": self.experiment_id,
            "turn_id": turn_id,
            "scenario": self.scenario_name,
            "speaker_slot": speaker_slot,
            "speaker_name": speaker.name,
            "speaker_model": speaker.model_slug,
            "responder_model": responder.model_slug,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": response_data["latency_ms"],
            "input_tokens": response_data["input_tokens"],
            "output_tokens": response_data["output_tokens"],
            "input_preview": input_message[:120],
            "content": response_data["content"],
            "finish_reason": response_data["finish_reason"],
            "is_refusal": response_data["is_refusal"],
            "system_prompt_snapshot": system_prompt_snapshot,
        }

    def save_logs(self, filepath: str) -> None:
        """Persist simulation logs to a JSONL file."""
        output_path = validate_non_empty_string(filepath, "filepath")
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        try:
            with open(output_path, "a", encoding="utf-8") as file_handle:
                for entry in self.logs:
                    file_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            logger.exception("Failed to save logs to %s", output_path)
            raise

        log_structured(
            logging.INFO,
            "logs_saved",
            filepath=output_path,
            entries_saved=len(self.logs),
        )
