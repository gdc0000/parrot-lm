# Stylized Pseudocode Map

Goal: describe the main algorithm with explicit traceability to concrete components.

## Algorithm A - Bootstrap and run simulation
- Component: `Application Entrypoint`
- File: `main.py`
- Functions: `main`, `initialize_infrastructure`, `configure_simulation_agents`, `execute_simulation`

```text
ALGORITHM A: RUN_SIMULATION_PIPELINE

INPUT:
  - config files + env vars

STEPS:
  1. setup_logging()
  2. config <- SimulationConfig.load()
  3. get_supabase_client(config.supabase_url, config.supabase_anon_key)
  4. agent_a_cfg <- AgentConfig(...persona_a..., ...generation params...)
  5. agent_b_cfg <- AgentConfig(...persona_b..., ...generation params...)
  6. orchestrator <- Orchestrator(agent_a_cfg, agent_b_cfg, scenario="simulation", api_key)
  7. buffered_logger <- SupabaseBufferedLogger(batch_size=config.batch_size)
  8. FOR log_entry IN orchestrator.run_simulation(config.num_turns, config.initial_message):
       buffered_logger.push(log_entry)
  9. ALWAYS buffered_logger.flush()

OUTPUT:
  - structured turn logs produced and uploaded in batches
```

## Algorithm B - Orchestrator turn loop and handoff
- Component: `Conversation Orchestrator`
- File: `parrotlm/orchestrator.py`
- Functions: `run_simulation`, `process_conversation_turns`, `_run_single_agent_turn`, `_create_log_entry`

```text
ALGORITHM B: ORCHESTRATE_A_B_TURNS

INPUT:
  - num_turns
  - initial_message

STATE:
  - last_message
  - agent_a, agent_b
  - agent_a_parameters, agent_b_parameters

STEPS:
  1. validate num_turns > 0
  2. last_message <- validate_non_empty_string(initial_message)
  3. log_simulation_start()
  4. REPEAT turn_index = 0..num_turns-1:
       a) (log_a, last_message, stop) <- _run_single_agent_turn(
            speaker=agent_a, responder=agent_b, input_message=last_message, params=agent_a_parameters
          )
       b) YIELD log_a
       c) IF stop THEN BREAK
       d) (log_b, last_message, stop) <- _run_single_agent_turn(
            speaker=agent_b, responder=agent_a, input_message=last_message, params=agent_b_parameters
          )
       e) YIELD log_b
       f) IF stop THEN BREAK
  5. log_simulation_completion(total_logs)

OUTPUT:
  - stream of normalized log entries (one per agent response)
```

## Algorithm C - Single agent turn (generate, normalize, stop condition)
- Component: `Turn Execution Unit`
- File: `parrotlm/orchestrator.py`
- Functions: `request_agent_generation`, `normalize_agent_payload`, `evaluate_stop_condition`, `_run_single_agent_turn`

```text
ALGORITHM C: EXECUTE_SINGLE_TURN

INPUT:
  - speaker, responder
  - input_message
  - generation_parameters
  - turn_index

STEPS:
  1. response_data <- speaker.generate_response(input_message, **generation_parameters)
  2. normalized <- normalize_response_data(response_data)
  3. log_entry <- _create_log_entry(..., input_message, normalized, ...)
  4. next_message <- normalized.content
  5. IF next_message is empty THEN next_message <- "..."
  6. should_stop <- evaluate_stop_condition(turn_index, speaker.name, normalized.is_refusal)
  7. RETURN (log_entry, next_message, should_stop)

OUTPUT:
  - one log entry + handoff message + stop flag
```

## Algorithm D - Context construction per model call
- Component: `Agent Context Manager`
- File: `parrotlm/agent.py`
- Functions: `generate_response`, `_build_request_messages`, `_prune_history`, `append_user_message`

```text
ALGORITHM D: BUILD_AND_MAINTAIN_CONTEXT_WINDOW

INPUT:
  - input_text
  - max_history_turns
  - agent local history

STEPS:
  1. append_user_message(input_text)
  2. relevant_history <- history without system message
  3. truncate relevant_history to last (max_history_turns * 2)
  4. IF relevant_history starts with assistant THEN drop first message
  5. messages_for_api <- [system_prompt] + relevant_history
  6. call LLM with messages_for_api
  7. extract metrics/content
  8. IF content exists THEN append assistant message
  9. _prune_history() with same bounded-window + role-alignment rules

OUTPUT:
  - response metrics
  - updated bounded local history
```

Context actually sent to the LLM:
- system prompt of the current agent
- recent bounded local history only
- latest incoming handoff as final `user` message

## Algorithm E - Buffered batch upload to Supabase
- Component: `Batch Persistence Logger`
- File: `parrotlm/supabase_logger.py`
- Functions: `SupabaseBufferedLogger.push`, `SupabaseBufferedLogger.flush`, `sanitize_log_entries`, `execute_batch_insert`, `verify_client_availability`

```text
ALGORITHM E: BUFFER_AND_FLUSH_TO_SUPABASE

INPUT:
  - incoming log_entry stream
  - batch_size

INIT:
  - (is_available, client, err) <- verify_client_availability()
  - buffer <- []

PUSH(log_entry):
  1. IF not is_available THEN RETURN
  2. buffer.append(log_entry)
  3. IF len(buffer) >= batch_size THEN FLUSH()

FLUSH():
  1. IF buffer empty OR not is_available THEN RETURN
  2. cleaned <- sanitize_log_entries(buffer)
  3. execute_batch_insert(client, cleaned)  # insert into session_logs
  4. buffer <- []

OUTPUT:
  - validated chunk inserts into Supabase table `session_logs`
```

Sanitization rule before insert:
- keep only schema-allowed fields (`_ALLOWED_COLUMNS`) to prevent batch failure on extra keys.
