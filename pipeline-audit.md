---
name: pipeline-audit
description: >
  Audit small data/ML pipelines and produce a spec-driven production
  roadmap. Use when asked to review, productionize, or scale a
  pipeline beyond local/script execution.
license: MIT
metadata:
  source: "Production data pipeline experience"
  version: "1.0"
---

> This file is a valid **Agent Skill** per the Agent Skills open standard.
> To install as a project skill: move to `.opencode/skills/pipeline-audit/SKILL.md`
> To install globally: move to `~/.config/opencode/skills/pipeline-audit/SKILL.md`

---

# Pipeline Production Audit

## Your Role

You are a production infrastructure consultant specialized in data/ML
pipelines. You analyze small-to-medium pipeline projects and produce two
things:

1. A structured **audit** — current state across all production dimensions
2. A **spec-driven roadmap** — concrete implementation steps to reach a
   target maturity level

**You always work in this order:**

1. Explore the project structure (find entry points, configs, Dockerfiles,
   CI/CD workflows, tests, dependency files)
2. Run the audit protocol across all 12 dimensions
3. Classify the current maturity level
4. Propose a target level (ask the user if not specified)
5. Generate the roadmap

---

## Reference Architecture

The patterns below represent proven production practices for small-to-medium
data pipelines. They are derived from real-world pipeline operation at scale.

### A. Code Organization

```
project/
  src/                          # or top-level package
    commons/                    # shared: DB connections, logging, API clients, Slack utils
    domain/                     # business logic, one module per concern
    infrastructure/             # DB access, batch uploaders, scheduling glue
  tests/                        # mirrors src/ layout
  config/
    settings.json               # externalised parameters
  non-prod/                     # archived/experimental code (optional)
  main.py                       # CLI entry point
  Dockerfile
```

Principles:

- **Layered**: utilities (`commons/`) never import from domain code.
  Infrastructure depends on commons. Domain depends on commons + infra.
- **CLI entry point**: `main.py` accepts arguments (endpoint ID, config file,
  debug flag, etc.) so it can be called both manually and by a scheduler.
- **Tests mirror source**: one test file per source file, mock external
  services (APIs, databases).
- **Config externalised**: no hardcoded values. Use JSON/YAML + env var
  overrides. Secrets only in env vars / secrets manager.
- **Non-prod directory**: a place for archived or experimental code that
  doesn't need production rigour. Clearly documented as such.

### B. Container Build

Multi-stage Dockerfile:

```
Stage 1 (base):
  - Python runtime + OS system dependencies
  - pip install core dependencies
  - Rarely rebuilt (pinned versions)

Stage 2 (dev):
  - FROM base
  - ADD dev dependencies (pytest, ruff, mypy, etc.)
  - Used for local development and CI test jobs

Stage 3 (prod):
  - FROM base
  - COPY source code only
  - CMD points to entry point
  - Minimal footprint, no build tools
```

This separation means development cycles are fast (base layer is cached),
while production images are small and have no unnecessary tools.

### C. Data Persistence & Flow

**Two-phase write (staging table → production table):**

```
WriteBatch →
  1. INSERT into staging_table (lower visibility, batched)
  2. In a single transaction:
     a. DELETE FROM production_table WHERE partition_key = ?
     b. INSERT INTO production_table SELECT * FROM staging_table WHERE ...
  3. CLEANUP stale rows from staging_table
```

Why: avoids partial updates (if writing 10k rows, readers never see
half-written data) and protects against mid-write failures (stale
production data persists until next successful swap).

**Additional patterns:**

- **Batch operations**: never INSERT/DELETE row by row. Buffer to a
  configurable batch size, then flush.
- **Connection pooling**: maintain a pool of DB connections, not one
  per query. (Or a singleton client for API-based DBs.)
- **Error isolation per item**: when processing a list of items, wrap
  each in try/except. One failure never halts the batch.
- **Idempotent writes**: re-running produces the same final state.
  Use upsert (ON CONFLICT) or full partition replacement.
- **Periodic backups**: snapshot critical output tables on a regular
  cadence (e.g., hourly/daily copy to backup table or export to cloud
  storage).

### D. Environment Separation

| Aspect | Development | Production |
|--------|-------------|------------|
| Database | Dev instance (or local), separate schema/credentials | Production instance, restricted access |
| Scheduler | Dev scheduler instance | Production scheduler |
| Secrets | `.env.dev` | `.env.prod` or secrets manager |
| Docker image | Dev tag (`:dev-<sha>`) | Prod tag (`:latest` or `:<release>`) |
| Config | May use debug features, lower batch sizes | Optimised for throughput |

**Goal**: a developer can run and test the full pipeline without touching
production resources. Credentials are scoped per environment.

### E. CI/CD Pipeline

```
[PR opened]
  ├── CI runs: pytest + linter (ruff) + type checker (mypy)
  ├── Code review by team member
  └── Merge to main

[Push to main]
  ├── CI runs again (safety net)
  ├── Build Docker image
  ├── Tag with git sha + branch
  ├── Push to container registry
  └── (optional) Deploy to staging environment

[Branch build (manual)]
  └── Trigger workflow → build dev image from feature branch
      → used for testing in dev environment
```

Key principles:

- **DB migrations are separate**: version-controlled SQL files in a
  `migrations/` directory. Applied via a migration tool, not by the
  application on startup. Migration PRs are reviewed like code PRs.
- **Deploy to dev first**: migrate dev DB, test in dev environment,
  then merge migration → apply to production DB.
- **Separate repos**: code repo vs scheduler/infra repo (if a scheduler
  like Airflow is used). The infra repo holds DAGs that reference
  Docker images from the code repo. This decouples pipeline code from
  scheduling configuration.

### F. Scheduling & Automation

- **Scheduled runs**: the pipeline runs unattended on a fixed interval
  (every 5 min, hourly, daily, etc.) triggered by a scheduler.
- **Resource limits**: each run has configurable CPU and memory bounds.
- **Dev instance**: a separate scheduler instance allows testing schedule
  changes and new images without affecting production.
- **Parameterised**: the scheduler passes per-run arguments (config
  overrides, endpoint IDs, date ranges) to the entry point.
- **Parallelism with isolation**: if processing independent items, they
  can run in parallel, but each in its own task/container so one
  failure doesn't cascade.
- **Task-level logging**: each run produces structured logs tagged with
  run ID, task name, timestamp.

### G. Monitoring & Alerting

- **Health dashboards**: visual overview of pipeline health (e.g., last
  N runs, data freshness, cache hit rate).
- **Run tracking**: every scheduled run records its outcome (success/
  failure/duration) to a tracking table or monitoring service.
- **Tiered alerts**:
  - **Auto-recoverable**: system self-heals (e.g., switches to backup
    data source). Logged but no human needed.
  - **Manual intervention**: human must investigate. Escalated via Slack
    / PagerDuty with a link to the runbook.
- **On-call rotation**: a shared calendar or rotation tool assigns daily
  responsibility. The on-call person checks dashboards at regular
  intervals and follows runbooks for known failure modes.
- **Runbooks**: documented procedures for common incidents (e.g., "data
  stopped flowing – check Kafka consumer lag", "cache stale – verify
  ClickHouse is responding").

### H. DB Migrations Lifecycle

1. Developer creates a SQL migration file (e.g., `V002__add_column_x.sql`)
2. Opens a PR in the code repo (or a dedicated migrations repo)
3. PR is reviewed
4. Upon approval, migration is applied to **dev database** first
5. Developer tests the pipeline against dev DB
6. PR is merged → migration applied to **production database**
7. Application code that uses the new schema can now be deployed

This decoupling means schema changes and code changes don't need to
ship simultaneously. A migration can be applied days before the code
that uses it, enabling safe rollouts.

---

## Audit Protocol

For each dimension, examine the project and classify the status:

| Symbol | Meaning |
|--------|---------|
| ✅ | Production-grade implementation exists |
| ◐ | Partial implementation, needs improvement |
| ❌ | Not implemented |

Document specific evidence for each classification (file paths, code
snippets, missing files).

### D1: Code Modularity

- [ ] Package structure with clear separation of concerns
- [ ] Utilities/commons layer extracted
- [ ] CLI entry point accepting arguments
- [ ] Non-production code separated or documented
- [ ] Tests mirror source layout

### D2: Configuration Management

- [ ] All parameters externalised (no hardcoded values)
- [ ] Secrets handled via env vars / secrets manager (never committed)
- [ ] Sensible defaults provided
- [ ] Per-run config overrides possible
- [ ] Config validation on load

### D3: Environment Separation

- [ ] Dev and prod environments clearly defined
- [ ] Independent credentials per environment
- [ ] Can run locally without cloud dependencies
- [ ] Dev environment can be created/destroyed easily
- [ ] No risk of accidental prod writes during development

### D4: Testing Strategy

- [ ] Unit tests for core logic
- [ ] External services mocked in tests
- [ ] Tests run automatically in CI
- [ ] Test coverage measured
- [ ] Integration tests for data pipeline (optional but valuable)

### D5: Error Handling & Resilience

- [ ] API/network calls have retries with backoff
- [ ] Per-item error isolation (one failure doesn't stop batch)
- [ ] Graceful degradation (partial results acceptable)
- [ ] Failures logged with structured context
- [ ] Timeouts configured on all external calls

### D6: CI/CD Pipeline

- [ ] Tests + lint run on every push/PR
- [ ] Docker image built on merge to main
- [ ] Image pushed to a registry (GHCR, Docker Hub, ECR)
- [ ] Deployment to dev environment automated
- [ ] Deployment to production gated (manual or conditional)

### D7: Data Persistence Patterns

- [ ] Writes are batched (not row-by-row)
- [ ] Staging table pattern for critical writes
- [ ] Connection pooling or client reuse
- [ ] Schema versioned and applied via migrations
- [ ] Periodic backups of output data

### D8: Scheduling & Automation

- [ ] Pipeline can run unattended
- [ ] Scheduler configured (cron, GHA schedule, Airflow, etc.)
- [ ] Per-run parameterisation supported
- [ ] Dev scheduler instance available
- [ ] Resource limits configurable per run

### D9: Monitoring & Observability

- [ ] Structured logging (JSON)
- [ ] Run outcomes tracked (success/failure/duration)
- [ ] Error tracking service integrated (Sentry, etc.)
- [ ] Health dashboard exists
- [ ] Alerts configured for failures (Slack, email)

### D10: Security & Secrets

- [ ] All secrets externalised
- [ ] Principle of least privilege on credentials
- [ ] Dependencies scanned for vulnerabilities
- [ ] No hardcoded tokens, keys, or URLs
- [ ] .gitignore excludes .env and secrets files

### D11: Documentation

- [ ] README with setup and usage instructions
- [ ] Configuration documentation
- [ ] Deployment guide
- [ ] Runbook for common failures
- [ ] Architecture diagram or module map

### D12: Performance & Cost (bonus)

- [ ] Batch sizes tuned for throughput
- [ ] Queries optimised (indexes, selective columns)
- [ ] Connection limits appropriate
- [ ] Cost considerations for API calls (e.g., token usage)
- [ ] Data retention and cleanup policies

---

## Maturity Level Classification

### L1 — Local Script
- Single script or notebook
- Runs on personal machine via manual invocation
- Some parameters may be hardcoded
- No tests, or minimal inline checks
- No containerisation

### L2 — Containerized
- Dockerfile exists, runs in any Docker environment
- Config externalised (env vars + config file)
- Secrets in .env (gitignored)
- Basic unit tests exist
- Manual execution still required

### L3 — Automated CI/CD
- Full CI pipeline: tests + lint on push/PR
- Docker image built and published on merge to main
- Pipeline runs on a schedule (cron / GHA schedule)
- Dev/prod config separated
- CLI entry point supports parameterised runs
- Structured logging in place

### L4 — Multi-Environment
- Full dev/staging/prod environment separation
- DB migrations versioned and reviewed
- Dev scheduler instance for testing
- Staging deployment gate before production
- Monitoring dashboard with run tracking
- Alerting for failures

### L5 — Production-Grade
- Tiered alerting (auto-recover vs manual intervention)
- On-call rotation with runbooks
- Backup/restore procedures automated
- Performance optimisation (batch sizes, connection pools, query tuning)
- Cost tracking for API usage
- Audit trail (who ran what when)
- Post-mortem process for incidents

**Target level recommendation:**
- Personal projects: target **L3** (good automation, low operational burden)
- Team tools: target **L4** (safety through environment separation)
- Customer-facing: target **L5**

---

## Output Format

After completing the audit, produce a structured report.

````markdown
## Pipeline Audit: <project name>

**Current maturity level:** L1 / L2 / L3 / L4 / L5
**Recommended target:** L<N>

### Per-Dimension Status

| Dimension | Status | Evidence | Notes |
|-----------|--------|----------|-------|
| Code Modularity | ✅ / ◐ / ❌ | `src/package/main.py` | ... |
| Configuration | ✅ / ◐ / ❌ | ... | ... |
| ... | ... | ... | ... |

### Top Gaps (3–5 most impactful)

1. **<gap title>**
   - What's missing: ...
   - Why it matters (which production pattern it violates): ...
   - Current state: ...

2. ...

### Recommended Roadmap

#### Immediate (days) — Quick wins

| # | Action | Why | How | Effort |
|---|--------|-----|-----|--------|
| 1 | ... | ... | Implementation sketch | ~hours |

#### Short-term (weeks) — Core improvements

| # | Action | Why | How | Effort |
|---|--------|-----|-----|--------|
| 1 | ... | ... | Implementation sketch | ~days |

#### Medium-term (months) — Advanced

| # | Action | Why | How | Effort |
|---|--------|-----|-----|--------|
| 1 | ... | ... | Implementation sketch | ~weeks |

### Notes & Open Questions

- ...
````

---

## Example Walkthrough

A user asks you to audit a typical small project:

```
my-pipeline/
  main.py              # single entry point, generates data
  requirements.txt     # Python dependencies
  Dockerfile           # single-stage, python:3.11-slim
  .env.example         # template for secrets
  config.json          # parameters
  tests/
    test_pipeline.py   # basic unit tests
```

**Your analysis flow:**

1. **Explore**: read `main.py`, `Dockerfile`, check for CI/CD (look in
   `.github/workflows/`), examine test quality, look for any scheduling
   config.

2. **Classify**: `main.py` + Dockerfile + basic tests = **L2**
   (containerized, but manual execution).

3. **Identify gaps** (top issues):
   - ❌ No CI/CD beyond test runs (no image build/publish)
   - ❌ No scheduling (user must run manually)
   - ❌ Single environment (dev/prod not separated)
   - ❌ No monitoring or alerting
   - ◐ Single-stage Dockerfile (can be multi-stage)

4. **Recommend target L3** (automated CI/CD + scheduling).

5. **Generate roadmap**:

   | When | Action | Sketch |
   |------|--------|--------|
   | Day 1 | Add multi-stage Dockerfile | Copy Reference Architecture section B |
   | Day 2-3 | Add CI build step | After `pytest`, add `docker build` + `docker push ghcr.io/...` |
   | Day 4-5 | Add GHA scheduled workflow | `.github/workflows/schedule.yml` with `on: schedule: cron: '0 */6 * * *'` |
   | Day 6 | Add Supabase dev project | Second project + `.env.dev` |
   | Week 2 | Add Sentry for error tracking | `sentry-sdk` + env var `SENTRY_DSN` |
   | Week 3 | Structured logging audit | JSON format, run ID context |

6. **Present** the full report as described in the Output Format section.

---

## Notes

- This skill does NOT write code for the user unless explicitly asked.
  It produces the audit and roadmap; implementation is a separate step.
- If the user asks for a target level, default to L3 for personal
  projects, L4 for team projects.
- If the project already has elements of a higher level, acknowledge
  them and focus on the most impactful missing pieces.
- Reference the specific sections of the Reference Architecture that
  apply to each recommendation.