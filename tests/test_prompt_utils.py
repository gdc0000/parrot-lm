import unittest

from parrotlm.prompt_utils import construct_system_prompt


class TestPromptUtils(unittest.TestCase):
    def test_construct_system_prompt_includes_persona_and_rules(self):
        persona = "A curious engineer"
        prompt = construct_system_prompt(persona)

        self.assertIn(persona, prompt)
        self.assertIn("DIALOGUE ONLY", prompt)
        self.assertIn("YOUR PERSONA", prompt)

