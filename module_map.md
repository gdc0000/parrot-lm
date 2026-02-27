# ParrotLM Module Map

## Overview
ParrotLM is organized by concern areas under `parrotlm/`:
- `agents`
- `configuration`
- `infrastructure`
- `orchestration`
- `validation`

The entrypoint `main.py` wires these concerns together.

## Folder-by-Concern Map

### Entrypoint
- `main.py`: Bootstraps configuration, starts orchestration, and coordinates logging/output flow.

### Agent
- `parrotlm/agents/__init__.py`: Agent package exports.
- `parrotlm/agents/agent.py`: LLM agent behavior (prompting, message history, model calls, retry behavior).

### Configuration
- `parrotlm/configuration/__init__.py`: Configuration package exports.
- `parrotlm/configuration/simulation_config.py`: Loads and validates runtime settings from env/JSON sources.
- `config/simulation.json`: Default/static simulation settings.

### Infrastructure
- `parrotlm/infrastructure/__init__.py`: Infrastructure package exports.
- `parrotlm/infrastructure/_logging.py`: Structured logging setup and helpers.
- `parrotlm/infrastructure/supabase_client.py`: Supabase client construction/caching.
- `parrotlm/infrastructure/supabase_logger.py`: Persists simulation/session logs to Supabase.

### Orchestration
- `parrotlm/orchestration/__init__.py`: Orchestration package exports.
- `parrotlm/orchestration/orchestrator.py`: Multi-turn conversation loop and per-turn event coordination.

### Validation
- `parrotlm/validation/__init__.py`: Validation package exports.
- `parrotlm/validation/_validators.py`: Input/value validation and safe type parsing.
- `parrotlm/validation/prompt_utils.py`: Persona/prompt construction rules and prompt-level constraints.

## Tests by Concern
- `tests/agents/test_agent.py`
- `tests/configuration/test_simulation_config.py`
- `tests/infrastructure/test_logging.py`
- `tests/infrastructure/test_supabase_client.py`
- `tests/infrastructure/test_supabase_logger.py`
- `tests/orchestration/test_orchestrator.py`
- `tests/validation/test_prompt_utils.py`
- `tests/validation/test_validators.py`
- `tests/application/test_main.py`
- `tests/README.md`

## High-Level Flow
1. `main.py` loads runtime configuration.
2. `orchestration/orchestrator.py` drives turn-by-turn interaction.
3. `agents/agent.py` handles model interaction for each agent turn.
4. `validation/*` enforces input and prompt correctness.
5. `infrastructure/*` records telemetry and uploads logs.
