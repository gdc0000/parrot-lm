# ParrotLM
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

ParrotLM is a Python framework for simulating and analyzing conversations between two LLM chatbots through a Streamlit UI. It supports custom personas, OpenRouter model slugs, live turn-by-turn execution, and built-in linguistic analysis.

## Features
- Two-chatbot conversation simulation with live streaming in the UI.
- Persona-driven system prompt construction with strict dialogue-only constraints.
- OpenRouter integration via the OpenAI Python client.
- Technical runtime controls from the sidebar:
  - turns per chatbot,
  - temperature per chatbot,
  - max tokens,
  - context window (history depth).
- Analysis tabs:
  - basic metrics (average latency and output tokens by model),
  - stylometric analysis with NLTK (token/sentence metrics and POS ratios),
  - custom lexicon categories (LIWC-style word counting),
  - CSV export of analyzed data.
- Browser-local persistence of run logs (`streamlit-local-storage`) with clear/reset support.

## Current Architecture
- `gui_app.py`: Streamlit entrypoint and tab composition.
- `parrotlm/ui/sidebar.py`: API key input and technical settings controls.
- `parrotlm/ui/chat_setup_tab.py`: chatbot setup and simulation execution flow.
- `parrotlm/orchestrator.py`: chatbot runtime, retry logic, log generation, optional JSONL save.
- `parrotlm/prompt_utils.py`: persona-to-system-prompt construction.
- `parrotlm/analysis_utils.py`: NLTK and custom lexicon analysis functions.
- `parrotlm/ui/session_state.py`: session state and local storage sync helpers.

## Requirements
- Python 3.10+
- OpenRouter API key (`OPENROUTER_API_KEY`)

Install dependencies:

```bash
pip install -r requirements.txt
```

## NLTK Setup
Stylometric analysis requires local NLTK resources. Install them once:

```bash
python -c "import nltk; [nltk.download(r) for r in ['punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng','universal_tagset']]"
```

## Configuration
Set your API key in one of these ways:
- Environment variable:
  - `OPENROUTER_API_KEY=...`
- `.env` file (copy from `.env.example`)
- Streamlit sidebar input field at runtime

## Run The App
```bash
python -m streamlit run gui_app.py
```

In the UI:
1. Configure model slugs and personas for Chatbot A and Chatbot B.
2. Set the initial message.
3. Tune technical settings in the sidebar.
4. Start the conversation and review analysis tabs.

## Data Behavior
- During normal UI usage, logs are stored in:
  - `st.session_state["all_logs"]`
  - browser local storage key `parrot_lm_logs`
- `Orchestrator.save_logs(filepath)` supports JSONL persistence for programmatic usage, but the Streamlit flow currently persists locally in browser storage by default.

## Tests
Run unit tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

See `tests/README.md` for coverage details by module.

## License
Licensed under Apache License 2.0. See `LICENSE`.

