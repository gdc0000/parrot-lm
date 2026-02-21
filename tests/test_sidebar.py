import os
from types import SimpleNamespace
from unittest.mock import patch

from parrotlm.ui import sidebar


class _FakeSidebar:
    def __init__(self, text_input_value, button_value, slider_values):
        self._text_input_value = text_input_value
        self._button_value = button_value
        self._slider_values = slider_values

    def header(self, *_args, **_kwargs):
        return None

    def text_input(self, *_args, **_kwargs):
        return self._text_input_value

    def button(self, *_args, **_kwargs):
        return self._button_value

    def slider(self, label, *_args, **_kwargs):
        return self._slider_values[label]

    def markdown(self, *_args, **_kwargs):
        return None


def test_apply_api_key_if_present_sets_env_var():
    with patch.dict(os.environ, {}, clear=True):
        sidebar._apply_api_key_if_present("test-key")

        assert os.environ["OPENROUTER_API_KEY"] == "test-key"


def test_apply_api_key_if_present_ignores_empty_value():
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "existing"}, clear=True):
        sidebar._apply_api_key_if_present("")

        assert os.environ["OPENROUTER_API_KEY"] == "existing"


def test_render_sidebar_returns_settings_and_clear_flag():
    fake_sidebar = _FakeSidebar(
        text_input_value="key-from-sidebar",
        button_value=True,
        slider_values={
            "Turns per Chatbot": 7,
            "Chatbot A Temperature": 0.7,
            "Chatbot B Temperature": 1.2,
            "Max Tokens": 1200,
            "Context Window (Turns)": 15,
        },
    )
    fake_st = SimpleNamespace(sidebar=fake_sidebar)

    with patch.dict(os.environ, {}, clear=True):
        with patch.object(sidebar, "st", fake_st):
            settings, clear_requested = sidebar.render_sidebar(default_turns=10)

        assert os.environ["OPENROUTER_API_KEY"] == "key-from-sidebar"

    assert clear_requested is True
    assert settings.num_turns == 7
    assert settings.temp_a == 0.7
    assert settings.temp_b == 1.2
    assert settings.max_tokens == 1200
    assert settings.context_window == 15


def test_render_sidebar_keeps_env_unchanged_when_api_key_blank():
    fake_sidebar = _FakeSidebar(
        text_input_value="",
        button_value=False,
        slider_values={
            "Turns per Chatbot": 3,
            "Chatbot A Temperature": 1.0,
            "Chatbot B Temperature": 1.0,
            "Max Tokens": 1000,
            "Context Window (Turns)": 20,
        },
    )
    fake_st = SimpleNamespace(sidebar=fake_sidebar)

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "existing-key"}, clear=True):
        with patch.object(sidebar, "st", fake_st):
            settings, clear_requested = sidebar.render_sidebar(default_turns=10)

        assert os.environ["OPENROUTER_API_KEY"] == "existing-key"

    assert clear_requested is False
    assert settings.num_turns == 3
