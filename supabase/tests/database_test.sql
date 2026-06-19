-- Start transaction and plan 11 tests.
BEGIN;
SELECT plan(11);

-- 1. Test existence of session_logs table
SELECT has_table('session_logs', 'Table "session_logs" should exist.');

-- 2. Test columns of session_logs table
SELECT has_column('session_logs', 'id', 'session_logs should have "id".');
SELECT col_is_pk('session_logs', 'id', 'session_logs "id" should be primary key.');
SELECT col_type_is('session_logs', 'experiment_id', 'text', 'experiment_id should be text.');
SELECT col_type_is('session_logs', 'turn_id', 'integer', 'turn_id should be integer.');
SELECT col_type_is('session_logs', 'latency_ms', 'double precision', 'latency_ms should be double precision.');

-- 3. Test existence of application_logs table
SELECT has_table('application_logs', 'Table "application_logs" should exist.');

-- 4. Test columns of application_logs table
SELECT has_column('application_logs', 'id', 'application_logs should have "id".');
SELECT col_is_pk('application_logs', 'id', 'application_logs "id" should be primary key.');
SELECT col_type_is('application_logs', 'level', 'text', 'level should be text.');
SELECT col_type_is('application_logs', 'timestamp', 'timestamp with time zone', 'timestamp should be timestamptz.');

-- Finish tests and rollback
SELECT * FROM finish();
ROLLBACK;
