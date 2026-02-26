# Test Suite Overview

This folder contains unit tests for the ParrotLM framework. The codebase maintains 100% test coverage for both happy and failure paths across all modules, ensuring high robustness and reliability.

## Structure

- `agents/test_agent.py`
- `application/test_main.py`
- `configuration/test_simulation_config.py`
- `infrastructure/test_logging.py`
- `infrastructure/test_supabase_client.py`
- `infrastructure/test_supabase_logger.py`
- `orchestration/test_orchestrator.py`
- `validation/test_prompt_utils.py`
- `validation/test_validators.py`

## Test Files

### `test_agent.py`
- **What it does:** Verifies the `Agent` class initialization, history bounding/pruning, request formatting, and metrics extraction. Specifically tests the `Tenacity` exponential backoff retry logic.
- **Why:** The agent handles direct interaction with the OpenRouter API. This ensures transient network failures are retried and token metrics are accurately captured without crashing the simulation.

### `test_logging.py`
- **What it does:** Verifies structured JSON logging utilities, event extraction, formatters (`HumanReadableFormatter` and `JsonLineFormatter`), and exception filtering logic.
- **Why:** Structured logging is critical for observability. These tests guarantee that logs are always serializable, properly formatted, and that sensitive or unrecoverable exceptions are properly handled.

### `test_main.py`
- **What it does:** End-to-end tests the `main.py` execution flow (initialization, configuration, execution, processing) using mocks to avoid network calls.
- **Why:** Ensures the top-level orchestration ties all the individual components together correctly and securely logs unhandled exceptions with their respective failed phase.

### `test_orchestrator.py`
- **What it does:** Tests the conversational ping-pong orchestration between two agents. Verifies turn limits, refusal detection, log entry generation, and parameter validation.
- **Why:** The Orchestrator manages the core multi-turn simulation loop. These tests verify the state machine transitions properly, stops when an agent refuses to answer, and emits properly structured logs.

### `test_prompt_utils.py`
- **What it does:** Verifies the injection of personas and strict dialogue-only formatting constraints into the system prompts.
- **Why:** Ensures the LLM receives the correct directives to act as a conversational partner rather than a narrator.

### `test_simulation_config.py`
- **What it does:** Tests environment variable loading, fallback defaults, type casting (integers and floats), and the optional `python-dotenv` initialization.
- **Why:** Ensures the application never crashes due to missing or malformed environment configuration, always falling back to safe defaults while logging the issue.

### `test_supabase_client.py`
- **What it does:** Tests the lazy initialization of the Supabase client singleton, credential resolution, and cache resetting.
- **Why:** Verifies the client handles missing credentials gracefully and accurately manages the HTTP connection pool for performance.

### `test_supabase_logger.py`
- **What it does:** Tests the batch upload process to Supabase, including checking client availability, strictly sanitizing log dictionaries to match the database schema, and handling insertion failures.
- **Why:** Ensures the application doesn't upload malformed records (which would cause PostgreSQL batch failures) and properly reports network or schema errors.

### `test_validators.py`
- **What it does:** Tests all input normalization and validation functions (strings, integers, generation parameters, response payloads).
- **Why:** Ensures invalid or missing required data throws immediate, highly-contextual errors before trickling down into confusing downstream bugs.

## Running the Tests

To run the entire test suite and verify everything passes:

```bash
python -m pytest
```

To run a quieter version of the test suite:
```bash
python -m pytest -q
```

## Design Notes
- **Isolation:** Tests extensively mock external APIs (OpenRouter, Supabase) using Python's `unittest.mock`. This keeps tests fast, deterministic, and free of network dependencies or quota consumption.
- **Comprehensive Coverage:** Every public function has at least one test for the happy path and one for a failure case, strictly adhering to our team-friendly engineering principles.
