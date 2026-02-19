# Test Suite Overview

This folder contains unit tests for the ParrotLM core package (`parrotlm`).

## Structure
- `test_simulation_config.py`
- `test_prompt_utils.py`
- `test_analysis_utils.py`
- `test_orchestrator.py`

## Test Files

### `test_simulation_config.py`
What it does:
- Verifies that core constants exist (`NUM_TURNS`, `DATA_DIR`).
- Verifies type and minimal validity (`NUM_TURNS > 0`, non-empty `DATA_DIR`).

Why:
- These constants drive defaults in the app.
- A missing or invalid constant would break startup or produce invalid runtime behavior.

### `test_prompt_utils.py`
What it does:
- Verifies `construct_system_prompt(persona)` includes:
  - the provided persona,
  - key instruction markers (for dialogue-only behavior).

Why:
- Prompt composition is critical for agent behavior.
- This catches regressions where persona injection or core constraints are accidentally removed.

### `test_analysis_utils.py`
What it does:
- Verifies category word counting (`count_custom_words`) is correct.
- Verifies log processing (`process_logs`) adds expected analysis columns.
- Uses mocking for `nltk` resource checks/downloads and for `analyze_text` where needed.

Why:
- Analysis output feeds charts and CSV export.
- Mocking keeps tests deterministic and avoids network/resource dependency for NLTK downloads.

### `test_orchestrator.py`
What it does:
- Verifies `Agent.generate_response` returns expected fields and token metadata.
- Verifies `Orchestrator.run_simulation(num_turns=1)` emits two log entries (one per agent turn).
- Mocks OpenAI client calls and injects a fake API key.

Why:
- Orchestration is the core runtime path.
- Mocking avoids external API calls, making tests fast and reliable while still validating conversation flow and log schema.

## Run Tests
From project root:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Design Notes
- Tests are unit-level and intentionally isolate external services.
- External dependencies (OpenRouter/OpenAI, NLTK downloads) are mocked to avoid flaky CI behavior.
- Coverage is focused on contract stability (inputs/outputs/fields), not UI rendering.
