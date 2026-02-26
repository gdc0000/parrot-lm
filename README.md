# ParrotLM
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

ParrotLM is a Python framework for simulating and analyzing multi-turn conversations between two LLM agents. It supports custom personas, OpenRouter model slugs, strict dialogue-only constraints, and automatically uploads structured interaction logs to a Supabase database.

## Features
- **Multi-turn conversation simulation**: Two agents automatically interact based on provided personas and an initial prompt.
- **OpenRouter integration**: Access a wide variety of models using a single API via the OpenAI Python client.
- **Supabase Cloud Logging**: Automatically sanitizes and uploads generated simulation logs directly to a `session_logs` Supabase table.
- **Robustness**: Built-in exponential backoff retries (via Tenacity) and robust validation/error handling.
- **Structured JSON Logging**: All application events are logged as machine-readable JSONlines for easy debugging and observability.
- **Clean Architecture**: Single-responsibility functions, explicit parameters, and 100% test coverage.

## Current Architecture
The system is organized into explicit layers:
- **Application Entrypoint**: `main.py`
- **Configuration Layer**: `parrotlm/configuration/simulation_config.py` + `config/simulation.json` + `.env`
- **Orchestration Layer**: `parrotlm/orchestration/orchestrator.py` (`AgentConfig`, `Orchestrator`)
- **Agent Layer**: `parrotlm/agents/agent.py` (single LLM agent with retry logic)
- **Infrastructure Layer**: `parrotlm/infrastructure/supabase_client.py` + `parrotlm/infrastructure/supabase_logger.py` + `parrotlm/infrastructure/_logging.py`
- **Validation Layer**: `parrotlm/validation/_validators.py` + `parrotlm/validation/prompt_utils.py`
- **Testing Layer**: `tests/` (9 test files, 100% coverage)

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

3. Configure your environment variables. Copy `.env.example` to `.env` and fill in your secrets:

```env
OPENROUTER_API_KEY=your_openrouter_key
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
```

4. Configure your simulation parameters in `config/simulation.json`:

```json
{
  "agents": {
    "agent_a": {
      "model": "openai/gpt-4o-mini",
      "persona": "Chief Technology Officer",
      "temperature": 1.0
    },
    "agent_b": {
      "model": "openai/gpt-4o-mini",
      "persona": "Financial Analyst",
      "temperature": 1.0
    }
  },
  "simulation": {
    "num_turns": 10,
    "initial_message": "What is your outlook on AI investment over the next 12 months?",
    "max_tokens": 1000,
    "context_window": 5
  }
}
```

5. Run the simulation:

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
