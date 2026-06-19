create table if not exists session_logs (
  id bigint generated always as identity primary key,
  experiment_id text not null,
  turn_id integer not null,
  scenario text,
  speaker_model text,
  responder_model text,
  timestamp timestamptz,
  latency_ms double precision,
  input_tokens integer,
  output_tokens integer,
  content text,
  finish_reason text,
  is_refusal boolean,
  system_prompt_snapshot text
);

create table if not exists application_logs (
  id bigint generated always as identity primary key,
  timestamp timestamptz not null,
  level text not null,
  logger_name text,
  module text,
  function_name text,
  line_number integer,
  event text,
  message text,
  context jsonb,
  exception text,
  process_id integer,
  thread_name text
);
