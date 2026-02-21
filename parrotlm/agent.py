"""Single LLM agent: model configuration, history window, retries, and API calls."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from openai import OpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from parrotlm._logging import is_retryable_exception, log_structured
from parrotlm._validators import get_openrouter_api_key, validate_non_empty_string

logger = logging.getLogger(__name__)


class Agent:
    """Represents a single LLM agent in a simulation."""

    def __init__(
        self,
        model_slug: str,
        system_prompt: str,
        name: str,
        max_history_turns: int = 20,
    ) -> None:
        self.model_slug = validate_non_empty_string(model_slug, "model_slug")
        self.system_prompt = validate_non_empty_string(system_prompt, "system_prompt")
        self.name = validate_non_empty_string(name, "name")

        if not isinstance(max_history_turns, int) or max_history_turns <= 0:
            raise ValueError("`max_history_turns` must be a positive integer.")
        self.max_history_turns = max_history_turns

        self.history: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=get_openrouter_api_key(),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(is_retryable_exception),
        reraise=True,
    )
    def generate_response(self, input_text: str, **kwargs: Any) -> Dict[str, Any]:
        """Generate one response from the model and return normalized metadata."""
        user_text = validate_non_empty_string(input_text, "input_text")
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
