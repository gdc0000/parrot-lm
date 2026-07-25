"""Single LLM agent: model configuration, history window, retries, and API calls."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from parrotlm.infrastructure._logging import is_retryable_exception, log_structured
from parrotlm.validation._validators import (
    validate_non_empty_string,
    validate_positive_int,
)

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(is_retryable_exception),
        reraise=True,
    )
    def _create_completion_stream(
        self, messages: List[Dict[str, str]], keyword_arguments: Dict[str, Any]
    ) -> Iterable[Any]:
        """Open a streaming completion request, retrying transient request-time errors.

        Only the stream *creation* is retried: once tokens have been emitted to the
        consumer, retrying a partially streamed response would duplicate text.
        """
        # We ignore type checking here because the openai library typings
        # require precise TypedDicts, but we're constructing dynamic dictionaries.
        return self.client.chat.completions.create(
            model=self.model_slug,
            messages=messages,  # type: ignore
            stream=True,
            stream_options={"include_usage": True},
            **keyword_arguments,
        )

    def execute_api_request(
        self, messages: List[Dict[str, str]], keyword_arguments: Dict[str, Any]
    ) -> Iterable[Any]:
        """Open a streaming request to the LLM API.

        Args:
            messages: The list of formatted message dictionaries.
            keyword_arguments: Additional arguments like temperature or max_tokens.

        Returns:
            The raw streaming response iterator.

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

        try:
            return self._create_completion_stream(messages, keyword_arguments)
        except Exception:
            logger.exception(
                "OpenRouter request failed for agent=%s model=%s.",
                self.name,
                self.model_slug,
            )
            raise

    def _consume_stream(
        self,
        stream: Iterable[Any],
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Tuple[str, str, Any]:
        """Accumulate streamed chunks into content, finish reason, and usage.

        Args:
            stream: The streaming response iterator.
            on_token: Optional callback invoked with each content chunk as it arrives.

        Returns:
            A tuple of (content, finish_reason, usage) where usage may be None.
        """
        content_parts: List[str] = []
        finish_reason = "unknown"
        usage = None

        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if choices:
                delta = getattr(choices[0], "delta", None)
                piece = getattr(delta, "content", None) if delta else None
                if piece:
                    content_parts.append(piece)
                    if on_token is not None:
                        on_token(piece)
                if choices[0].finish_reason:
                    finish_reason = choices[0].finish_reason
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage

        return "".join(content_parts).strip(), finish_reason, usage

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

    def generate_response(
        self,
        input_text: str,
        on_token: Optional[Callable[[str], None]] = None,
        **keyword_arguments: Any,
    ) -> Dict[str, Any]:
        """Generate one streamed response from the model and return normalized metadata.

        Args:
            input_text: The incoming message text.
            on_token: Optional callback invoked with each token chunk as it streams in.
            **keyword_arguments: Additional parameters for the generation API (e.g. max_tokens).

        Returns:
            A dictionary of parsed metrics including content, tokens, and finish reason.

        Note:
            Mid-stream failures are not retried: retrying after tokens have already
            been emitted would duplicate text for streaming consumers.
        """
        self.append_user_message(input_text)
        messages = self._build_request_messages()

        start_time = time.time()
        stream = self.execute_api_request(messages, keyword_arguments)
        content, finish_reason, usage = self._consume_stream(stream, on_token)
        latency_ms = (time.time() - start_time) * 1000

        if usage is None:
            logger.warning(
                "Provider returned no usage data for agent=%s model=%s; recording 0 tokens.",
                self.name,
                self.model_slug,
            )

        metrics = {
            "content": content,
            "latency_ms": latency_ms,
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
            "finish_reason": finish_reason,
            "is_refusal": not content or finish_reason == "content_filter",
        }

        self.update_conversation_history(content)

        return metrics
