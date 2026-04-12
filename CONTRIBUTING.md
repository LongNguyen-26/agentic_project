# CONTRIBUTING

## 1. Purpose
This document explains how to develop, test, and extend the current agent safely.

Goals:
- keep competition runtime stable
- preserve typed contracts and graph behavior
- keep docs/config/tests synchronized with code changes

## 2. Local Development Setup

### 2.1 Prerequisites
- Python 3.12+
- uv installed (recommended)

### 2.2 Install
From repository root:

```bash
uv sync --frozen --extra dev
```

### 2.3 Environment configuration
Create `.env` from `.env.example` and set at minimum:
- `COMPETITION_BASE_URL`
- `API_KEY`
- `OPENAI_API_KEY`

### 2.4 Quick runtime sanity check

```bash
uv run python -m devday_agent.main
```

Expected early log sequence includes auth/fetch lifecycle lines.

## 3. Test Workflow

Run all tests:

```bash
uv run pytest src/tests -q
```

Run focused suites:

```bash
uv run pytest src/tests/test_outer_loop.py -q
uv run pytest src/tests/test_inner_loop_verifiability.py -q
uv run pytest src/tests/test_document_parser.py -q
uv run pytest src/tests/test_llm_client_overflow.py -q
uv run pytest src/tests/test_planning_hints.py -q
uv run pytest src/tests/test_context_manager.py -q
uv run pytest src/tests/test_rag_engine.py -q
```

Rule: code changes in nodes/clients/tools should include or update tests in `src/tests`.

## 4. Codebase Conventions

- Source package path: `src/devday_agent`.
- Prefer absolute imports from `devday_agent.*`.
- Keep docstrings/comments/log messages in English.
- Maintain explicit return type hints for node/helper functions.
- Keep state/schema typing strict and aligned with pydantic models.
- Avoid introducing unused helper paths and dead fallback code.

## 5. Extending for New Task Types

When adding a new task type beyond `question-answering` and `folder-organisation`, update in this order.

1. Schema contracts
- `src/devday_agent/models/llm_schemas.py`
- extend `TaskClassification.task_type`
- add typed action/verification response models

2. Prompt layer
- `src/devday_agent/agent/prompts/sys_prompts.py`
- `src/devday_agent/agent/prompts/user_prompt.py`
- add `build_<task>_action_prompt(...)` and `build_<task>_verification_prompt(...)`

3. Outer-loop classification and planning
- `src/devday_agent/agent/nodes/outer_loop.py`
- extend rule-based fast path in `fetch_task_node`
- keep LLM classification fallback for ambiguous prompts

4. Inner-loop action + verification
- `src/devday_agent/agent/nodes/inner_loop.py`
- add `_generate_<task>_action(...)` and `_verify_<task>(...)`
- wire dispatch in action and verification nodes

5. Routing/graph changes (only if needed)
- `src/devday_agent/agent/nodes/router.py`
- `src/devday_agent/agent/graph.py`
- add new branch only when current flow cannot represent task requirements

6. Tests
- add task coverage in `src/tests` for:
  - classification
  - action schema
  - verification retry/pass
  - submission payload shape

## 6. Operational Troubleshooting

### 6.1 LLM context overflow
Symptoms:
- `maximum context length`
- `context_length_exceeded`
- `prompt is too long`

Current safeguards:
- message trimming while preserving system + latest user context
- retry with backoff/jitter

Checks:
1. reduce oversized prompt/context blocks
2. tune token budgets in `.env`
3. run `uv run pytest src/tests/test_llm_client_overflow.py -q`

### 6.2 Vision OCR/tool issues
Symptoms:
- image OCR failures or empty vision observations

Checks:
1. verify `OPENAI_API_KEY`
2. verify `MODEL_NAME` supports image input
3. inspect `storage/agent.log` for parser/vision warnings

### 6.3 API auth/rate-limit instability
Symptoms:
- frequent 401/429/5xx responses

Checks:
1. validate `COMPETITION_BASE_URL` and `API_KEY`
2. tune `HTTP_MAX_RETRIES`, `HTTP_BACKOFF_SECONDS`, `HTTP_TIMEOUT_SECONDS`
3. check checkpoint behavior in `storage/session_checkpoint.json`

## 7. Documentation Update Policy

When behavior/config/commands change, update all impacted docs in the same change set:
- `README.md`
- `ARCHITECTURE.md`
- `CONTRIBUTING.md`
- `.env.example` (if env keys/defaults change)

Do not leave docs in a partially migrated state.

## 8. Security and Secrets

- Never commit real keys/tokens.
- Keep `.env` local only.
- Sanitize logs/checkpoints before sharing artifacts.
