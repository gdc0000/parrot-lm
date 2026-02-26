"""Single LLM agent: model configuration, history window, retries, and API calls."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from openai.types.chat import ChatCompletion
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from parrotlm.infrastructure._logging import is_retryable_exception, log_structured
from parrotlm.validation._validators import validate_non_empty_string, validate_positive_int

logger = logging.getLogger(__name__)


class Agent:
    """Represents a single LLM agent in a simulation.

    Manages conversation history, formats API requests, and handles transient
    network errors using exponential backoff retries.
    """

    def __init__(
        self,
        model_slug: str,
        system_prompt: str,
        name: str,
        api_key: str,
        max_history_turns: int = 20,
    ) -> None:
        """Initialize the agent with its configuration and history.

        Args:
            model_slug: The specific model string to use (e.g., 'openai/gpt-4o').
            system_prompt: The persona and instructions for this agent.
            name: A human-readable identifier for logging.
            api_key: The OpenRouter API key.
            max_history_turns: How many turns of conversation history to retain.

        Raises:
            ValueError: If string arguments are empty or max_history_turns is not positive.
        """
        self.model_slug = validate_non_empty_string(model_slug, "model_slug")
        self.system_prompt = validate_non_empty_string(system_prompt, "system_prompt")
        self.name = validate_non_empty_string(name, "name")

        self.max_history_turns = validate_positive_int(
            max_history_turns, "max_history_turns", default=20
        )

        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def _build_request_messages(self) -> List[Dict[str, str]]:
        """Build a bounded, role-aligned message list for the next API request."""
        maximum_messages = self.max_history_turns * 2
        relevant_history = self.history[1:]
        if len(relevant_history) > maximum_messages:
            relevant_history = relevant_history[-maximum_messages:]

        # OpenRouter's API often strictly enforces alternating user/assistant messages.
        # If our history slice happens to start with an assistant message, we drop it
        # so the model sees a valid user-led exchange sequence (plus the current user
        # message at the end).
        if relevant_history and relevant_history[0].get("role") == "assistant":
            relevant_history = relevant_history[1:]

        return [{"role": "system", "content": self.system_prompt}] + relevant_history

    def _prune_history(self) -> None:
        """Keep local history bounded and role-aligned to match the configured window."""
        maximum_messages = self.max_history_turns * 2
        relevant_history = self.history[1:]
        if len(relevant_history) <= maximum_messages:
            return

        relevant_history = relevant_history[-maximum_messages:]
        if relevant_history and relevant_history[0].get("role") == "assistant":
            relevant_history = relevant_history[1:]
        self.history = [self.history[0]] + relevant_history

    def append_user_message(self, input_text: str) -> None:
        """Validate and append the user's input to the conversation history.

        Args:
            input_text: The user message string to append.

        Raises:
            ValueError: If input_text is empty.
        """
        user_text = validate_non_empty_string(input_text, "input_text")
        self.history.append({"role": "user", "content": user_text})

    def execute_api_request(
        self, messages: List[Dict[str, str]], keyword_arguments: Dict[str, Any]
    ) -> Tuple[ChatCompletion, float]:
        """Send the formatted messages to the LLM API and measure latency.

        Args:
            messages: The list of formatted message dictionaries.
            keyword_arguments: Additional arguments like temperature or max_tokens.

        Returns:
            A tuple containing the raw API response object and the latency in milliseconds.

        Raises:
            Exception: If the API request fails due to network or authentication issues.
        """
        log_structured(
            logging.DEBUG,
            "agent_request_context",
            agent=self.name,
            model=self.model_slug,
            history_total_messages=len(self.history) - 1,
            messages_sent=len(messages) - 1,
            roles_sent=[message.get("role", "unknown") for message in messages[1:]],
        )

        start_time = time.time()
        try:
            # We ignore type checking here because the openai library typings
            # require precise TypedDicts, but we're constructing dynamic dictionaries.
            response = self.client.chat.completions.create(
                model=self.model_slug,
                messages=messages,  # type: ignore
                **keyword_arguments,
            )
        except Exception:
            logger.exception(
                "OpenRouter request failed for agent=%s model=%s.",
                self.name,
                self.model_slug,
            )
            raise

        latency_ms = (time.time() - start_time) * 1000
        return response, latency_ms

    def extract_response_metrics(
        self, response: Any, latency_ms: float
    ) -> Dict[str, Any]:
        """Extract content, usage, and termination reasons from the raw API response.

        Args:
            response: The raw API response object.
            latency_ms: The recorded latency in milliseconds.

        Returns:
            A dictionary containing the parsed metrics.

        Raises:
            RuntimeError: If the response lacks required fields like choices.
        """
        choices = getattr(response, "choices", None) or []
        if not choices or not getattr(choices[0], "message", None):
            raise RuntimeError("Model response is missing message choices.")

        content = (choices[0].message.content or "").strip()
        finish_reason = choices[0].finish_reason or "unknown"
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        is_refusal = not content or finish_reason == "content_filter"

        return {
            "content": content,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "finish_reason": finish_reason,
            "is_refusal": is_refusal,
        }

    def update_conversation_history(self, content: str) -> None:
        """Append the assistant's reply (if any) and prune history to window limits.

        Args:
            content: The text generated by the assistant.
        """
        if content:
            self.history.append({"role": "assistant", "content": content})
            self._prune_history()
        else:
            self._prune_history()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(is_retryable_exception),
        reraise=True,
    )
    def generate_response(
        self, input_text: str, **keyword_arguments: Any
    ) -> Dict[str, Any]:
        """Generate one response from the model and return normalized metadata.

        Args:
            input_text: The incoming message text.
            **keyword_arguments: Additional parameters for the generation API (e.g. max_tokens).

        Returns:
            A dictionary of parsed metrics including content, tokens, and finish reason.
        """
        self.append_user_message(input_text)
        messages = self._build_request_messages()

        response, latency_ms = self.execute_api_request(messages, keyword_arguments)
        metrics = self.extract_response_metrics(response, latency_ms)

        self.update_conversation_history(metrics["content"])

        return metrics

