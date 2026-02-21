"""Core orchestration logic for running agent-to-agent conversations."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


def _log_structured(level: int, event: str, **context: Any) -> None:
    """Log one event with machine-readable context for easier debugging."""
    try:
        context_json = json.dumps(context, sort_keys=True, default=str)
    except (TypeError, ValueError):
        context_json = str(context)
    logger.log(level, "%s | %s", event, context_json)


def _is_retryable_exception(exception: BaseException) -> bool:
    """Retry transient failures, but not local validation errors."""
    # Type/Value errors usually indicate bad caller input and will not succeed on retry.
    return not isinstance(exception, (TypeError, ValueError))


def _get_openrouter_api_key() -> str:
    """Resolve the OpenRouter API key from environment or .env file."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        return api_key

    # Load .env lazily so normal env-based deployments do not pay this cost on every import.
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables or .env file.")
    return api_key


def _validate_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{field_name}` must be a non-empty string.")
    return value.strip()


def _validate_positive_int(value: Any, field_name: str, default: int) -> int:
    """Validate an optional positive integer value with fallback default."""
    resolved = default if value is None else value
    if not isinstance(resolved, int) or resolved <= 0:
        raise ValueError(f"`{field_name}` must be a positive integer.")
    return resolved


def _validate_generation_params(params: Any, field_name: str) -> Dict[str, Any]:
    """Validate optional per-agent model generation parameters."""
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise TypeError(f"`{field_name}` must be a dictionary.")
    return params


def _normalize_response_data(response_data: Any) -> Dict[str, Any]:
    """Validate and normalize one agent response payload."""
    if not isinstance(response_data, dict):
        raise TypeError("`response_data` must be a dictionary.")

    required_fields = [
        "content",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "finish_reason",
        "is_refusal",
    ]
    missing_fields = [field for field in required_fields if field not in response_data]
    if missing_fields:
        missing_csv = ", ".join(missing_fields)
        raise KeyError(f"Missing response fields: {missing_csv}")

    content_value = str(response_data["content"] or "").strip()
    normalized_response = {
        "content": content_value,
        "latency_ms": float(response_data["latency_ms"]),
        "input_tokens": int(response_data["input_tokens"]),
        "output_tokens": int(response_data["output_tokens"]),
        "finish_reason": str(response_data["finish_reason"] or "unknown"),
        "is_refusal": bool(response_data["is_refusal"]),
    }
    return normalized_response


class Agent:
    """Represents a single LLM agent in a simulation."""

    def __init__(
        self,
        model_slug: str,
        system_prompt: str,
        name: str,
        max_history_turns: int = 20,
    ) -> None:
        self.model_slug = _validate_non_empty_string(model_slug, "model_slug")
        self.system_prompt = _validate_non_empty_string(system_prompt, "system_prompt")
        self.name = _validate_non_empty_string(name, "name")

        if not isinstance(max_history_turns, int) or max_history_turns <= 0:
            raise ValueError("`max_history_turns` must be a positive integer.")
        self.max_history_turns = max_history_turns

        self.history: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=_get_openrouter_api_key(),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_retryable_exception),
        reraise=True,
    )
    def generate_response(self, input_text: str, **kwargs: Any) -> Dict[str, Any]:
        """Generate one response from the model and return normalized metadata."""
        user_text = _validate_non_empty_string(input_text, "input_text")
        self.history.append({"role": "user", "content": user_text})

        # One turn contributes two messages (user + assistant), so keep a bounded sliding window.
        max_messages = self.max_history_turns * 2
        # System prompt is always reconstructed explicitly below to avoid duplicated system entries.
        relevant_history = self.history[1:]
        if len(relevant_history) > max_messages:
            relevant_history = relevant_history[-max_messages:]
        messages = [{"role": "system", "content": self.system_prompt}] + relevant_history

        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_slug,
                messages=messages,
                **kwargs,
            )
        except Exception:
            logger.exception("OpenRouter request failed for %s.", self.name)
            raise

        latency_ms = (time.time() - start_time) * 1000

        choices = getattr(response, "choices", None) or []
        if not choices or not getattr(choices[0], "message", None):
            raise RuntimeError("Model response is missing message choices.")

        content = (choices[0].message.content or "").strip()
        finish_reason = choices[0].finish_reason or "unknown"
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        is_refusal = not content or finish_reason == "content_filter"

        if content:
            self.history.append({"role": "assistant", "content": content})

        return {
            "content": content,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "finish_reason": finish_reason,
            "is_refusal": is_refusal,
        }


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
        self.scenario_name = _validate_non_empty_string(scenario_name, "scenario_name")

        if not isinstance(agent_a_config, dict) or not isinstance(agent_b_config, dict):
            raise TypeError("`agent_a_config` and `agent_b_config` must be dictionaries.")

        max_history_turns_a = _validate_positive_int(
            agent_a_config.get("max_history_turns"),
            "agent_a_config['max_history_turns']",
            default=20,
        )
        max_history_turns_b = _validate_positive_int(
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
        self.agent_a_params = _validate_generation_params(
            agent_a_config.get("params"),
            "agent_a_config['params']",
        )
        self.agent_b_params = _validate_generation_params(
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
        last_message = _validate_non_empty_string(initial_message, "initial_message")

        _log_structured(
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

        _log_structured(
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
            normalized_response_data = _normalize_response_data(response_data)
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
        )
        self.logs.append(log_entry)

        # Keep a non-empty handoff message so the next agent receives valid input even on blank output.
        next_message = normalized_response_data["content"] or "..."
        should_stop = bool(normalized_response_data["is_refusal"])
        if should_stop:
            _log_structured(
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
    ) -> Dict[str, Any]:
        """Create a normalized log dictionary for one model output."""
        return {
            "experiment_id": self.experiment_id,
            "turn_id": turn_id,
            "scenario": self.scenario_name,
            "speaker_model": speaker.model_slug,
            "responder_model": responder.model_slug,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_ms": response_data["latency_ms"],
            "input_tokens": response_data["input_tokens"],
            "output_tokens": response_data["output_tokens"],
            "content": response_data["content"],
            "finish_reason": response_data["finish_reason"],
            "is_refusal": response_data["is_refusal"],
            "system_prompt_snapshot": system_prompt_snapshot,
        }

    def save_logs(self, filepath: str) -> None:
        """Persist simulation logs to a JSONL file."""
        output_path = _validate_non_empty_string(filepath, "filepath")
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

        _log_structured(
            logging.INFO,
            "logs_saved",
            filepath=output_path,
            entries_saved=len(self.logs),
        )
