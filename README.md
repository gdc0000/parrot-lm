# ParrotLM
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

ParrotLM is a Python framework for simulating and analyzing multi-turn conversations between two LLM agents. It supports custom personas, OpenRouter model slugs, strict dialogue-only constraints, and automatically uploads structured interaction logs to a Supabase database.

The codebase is built on **team-friendly principles**: simple over clever, readable over efficient, and optimized for the next developer. Every function has a single responsibility, explanatory naming, comprehensive docstrings, context-rich error handling, and full unit test coverage.

## Features
- **Multi-turn conversation simulation**: Two agents automatically interact based on provided personas and an initial prompt.
- **OpenRouter integration**: Access a wide variety of models using a single API via the OpenAI Python client.
- **Supabase Cloud Logging**: Automatically sanitizes and uploads generated simulation logs directly to a `session_logs` Supabase table.
- **Robustness**: Built-in exponential backoff retries (via Tenacity) and robust validation/error handling.
- **Structured JSON Logging**: All application events are logged as machine-readable JSONlines for easy debugging and observability.
- **Clean Architecture**: Single-responsibility functions, explicit parameters, and 100% test coverage.

## Current Architecture
- `main.py`: The main entrypoint. Initializes infrastructure, configures agents, executes the simulation, and uploads results.
- `parrotlm/agent.py`: Single LLM agent managing model configuration, history windows, retries, and API calls.
- `parrotlm/orchestrator.py`: Manages the simulation run, orchestrating the back-and-forth between two agents and generating structured logs.
- `parrotlm/prompt_utils.py`: Constructs system prompts with strict constraints (dialogue only, zero narration).
- `parrotlm/simulation_config.py`: Environment-based configuration loader.
- `parrotlm/supabase_client.py` & `parrotlm/supabase_logger.py`: Supabase client singleton and batch log insertion logic.
- `parrotlm/_logging.py` & `parrotlm/_validators.py`: Core utilities for structured logging and payload validation.

## Requirements
- Python 3.10+
- OpenRouter API key (`OPENROUTER_API_KEY`)
- Supabase Project URL and Anon Key (`SUPABASE_URL`, `SUPABASE_ANON_KEY`)

## Quickstart
1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

2. Install project dependencies:

```bash
pip install -r requirements.txt
```

3. Configure your environment variables. Copy `.env.example` to `.env` and fill in your credentials and simulation parameters:

```env
OPENROUTER_API_KEY=your_openrouter_key
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key

MODEL_A=openai/gpt-4o-mini
MODEL_B=openai/gpt-4o-mini
PERSONA_A="Chief Technology Officer"
PERSONA_B="Financial Analyst"
NUM_TURNS=10
INITIAL_MESSAGE="What is your outlook on AI investment over the next 12 months?"
MAX_TOKENS=1000
TEMPERATURE_A=1.0
TEMPERATURE_B=1.0
CONTEXT_WINDOW=5
```

4. Run the simulation:

```bash
python main.py
```

Logs will be output to your console and `logs/parrotlm.log` locally, and the final conversation turns will be uploaded to your Supabase `session_logs` table.

## Tests
The codebase maintains 100% test coverage for both happy and failure paths. Run the unit test suite via `pytest`:

```bash
python -m pytest
```

To run a quieter version of the tests:
```bash
python -m pytest -q
```

## License
Licensed under Apache License 2.0. See `LICENSE`.
