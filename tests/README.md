# Test Suite Overview

This folder contains unit tests for the ParrotLM core package (`parrotlm`).

## Structure
- `test_simulation_config.py`
- `test_prompt_utils.py`
- `test_analysis_utils.py`
- `test_orchestrator.py`
- `test_session_state.py`
- `test_sidebar.py`
- `test_chat_setup_tab.py`
- `test_analysis_tabs.py`
- `test_integration_pipeline.py`

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
- Verifies dataframe input validation for `process_logs` and `process_custom_lexicon`.
- Uses mocking for `nltk` resource checks/downloads and for `analyze_text` where needed.

Why:
- Analysis output feeds charts and CSV export.
- Mocking keeps tests deterministic and avoids network/resource dependency for NLTK downloads.

### `test_orchestrator.py`
What it does:
- Verifies `Agent.generate_response` returns expected fields and token metadata.
- Verifies `Orchestrator.run_simulation(num_turns=1)` emits two log entries (one per agent turn).
- Verifies distinct `max_history_turns` are applied independently to Agent A and Agent B.
- Verifies invalid non-dictionary `params` are rejected early with a clear error.
- Mocks OpenAI client calls and injects a fake API key.

Why:
- Orchestration is the core runtime path.
- Mocking avoids external API calls, making tests fast and reliable while still validating conversation flow and log schema.

### `test_session_state.py`
What it does:
- Verifies `initialize_session_state` sets required defaults when storage is empty.
- Verifies `initialize_session_state` correctly loads persisted logs from local storage.
- Verifies `clear_local_data` clears both local storage and in-memory dataframe.
- Verifies `append_and_persist_logs` appends rows and persists merged records.

Why:
- Session-state and local persistence are critical to consistent UI behavior.
- These tests protect against regressions in log initialization, reset, and persistence flow.

### `test_sidebar.py`
What it does:
- Verifies API-key environment synchronization behavior.
- Verifies `render_sidebar` maps Streamlit control values into `TechnicalSettings` correctly.
- Verifies blank API key input does not overwrite an existing env key.

Why:
- Sidebar values feed runtime orchestration parameters directly.
- These tests catch configuration regressions before they impact live simulation runs.

### `test_chat_setup_tab.py`
What it does:
- Verifies scenario naming behavior.
- Verifies simulation execution success/failure handling in `_execute_simulation`.
- Verifies log persistence path converts simulation logs into a dataframe before storage sync.

Why:
- Chat setup is the main user workflow entrypoint.
- These tests ensure API/runtime failures are surfaced safely and success paths persist data as expected.

### `test_analysis_tabs.py`
What it does:
- Verifies custom lexicon normalization behavior.
- Verifies aggregate metric computation used by charts.

Why:
- Analysis output powers user-facing comparisons and exported CSV data.
- Helper-level tests keep analysis transformations stable during refactors.

### `test_integration_pipeline.py`
What it does:
- Runs an integration-style pipeline from orchestrator output -> session persistence -> analysis processing.
- Uses mocked OpenAI responses to keep the flow deterministic.

Why:
- Confirms cross-module contracts work together, not only in isolated unit tests.
- Provides a higher-confidence regression safety net for end-to-end data flow.

## Run Tests
From project root:

```bash
python -m pytest -q
```

## Design Notes
- Most tests are unit-level and intentionally isolate external services.
- Integration-style coverage is included where feasible (see `test_integration_pipeline.py`).
- External dependencies (OpenRouter/OpenAI, NLTK downloads) are mocked to avoid flaky CI behavior.
- Coverage is focused on contract stability (inputs/outputs/fields), not UI rendering.
