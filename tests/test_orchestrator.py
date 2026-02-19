import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from parrotlm.orchestrator import Agent, Orchestrator


class _FakeCompletions:
    def create(self, model, messages, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="mocked reply"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


class _FakeOpenAIClient:
    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


class TestOrchestrator(unittest.TestCase):
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False)
    @patch("parrotlm.orchestrator.OpenAI", side_effect=_FakeOpenAIClient)
    def test_agent_generate_response_returns_expected_fields(self, _mock_openai):
        agent = Agent(
            model_slug="fake/model",
            system_prompt="You are concise.",
            name="Agent A",
            max_history_turns=5,
        )

        response = agent.generate_response("hello")

        self.assertEqual(response["content"], "mocked reply")
        self.assertEqual(response["finish_reason"], "stop")
        self.assertEqual(response["input_tokens"], 10)
        self.assertEqual(response["output_tokens"], 5)
        self.assertFalse(response["is_refusal"])

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False)
    @patch("parrotlm.orchestrator.OpenAI", side_effect=_FakeOpenAIClient)
    def test_run_simulation_emits_two_entries_for_one_turn(self, _mock_openai):
        agent_a_config = {
            "model": "fake/model-a",
            "system_prompt": "Persona A",
            "user_persona_snapshot": "Persona A",
        }
        agent_b_config = {
            "model": "fake/model-b",
            "system_prompt": "Persona B",
            "user_persona_snapshot": "Persona B",
        }

        orchestrator = Orchestrator(agent_a_config, agent_b_config, scenario_name="test")
        logs = list(orchestrator.run_simulation(num_turns=1, initial_message="Hi"))

        self.assertEqual(len(logs), 2)
        self.assertIn("content", logs[0])
        self.assertIn("system_prompt_snapshot", logs[0])

