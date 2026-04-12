# CONTRIBUTING

## 👋 Purpose
This guide is for developers receiving or extending the project.

Goal: keep the agent stable for competition runtime while allowing safe extension.

## ✅ Development Setup
1. Create environment and install dependencies.
2. Copy `.env.example` to `.env`.
3. Fill required keys: `COMPETITION_BASE_URL`, `API_KEY`, `OPENAI_API_KEY`.
4. Run `python main.py` and verify auth/fetch logs.

## 🧪 Testing
Run all unit tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Run focused suites:

```bash
python -m unittest tests.test_outer_loop
python -m unittest tests.test_inner_loop_verifiability
python -m unittest tests.test_document_parser
python -m unittest tests.test_llm_client_overflow
python -m unittest tests.test_planning_hints
python -m unittest tests.test_context_manager
```

## 🧩 Extending the Agent for New Task Types
If the organizer introduces a new task type (beyond `folder-organisation` and `question-answering`), update in this order:

1. Schema and type contract
- `models/llm_schemas.py`
- Extend `TaskClassification.task_type` literal.
- Add/adjust structured response model for the new task.

2. Classification prompts and prompt builders
- `agent/prompts/sys_prompts.py`
- Update `SYS_CLASSIFY_TASK` with explicit definition for the new type.
- Add new system prompts for action and verification if needed.
- `agent/prompts/user_prompt.py`
- Add `build_<new_type>_action_prompt(...)` and `build_<new_type>_verification_prompt(...)`.

3. Outer loop routing/classification
- `agent/nodes/outer_loop.py`
- Extend rule-based keyword fast-path in `fetch_task_node`.
- Keep LLM fallback classification as final fallback.

4. Inner loop execution logic
- `agent/nodes/inner_loop.py`
- Add `_generate_<new_type>_action(...)`.
- Add `_verify_<new_type>(...)`.
- Dispatch in `action_generation_node` and `verifiability_node`.

5. Graph routing if the new type needs a different processing path
- `agent/graph.py`
- `agent/nodes/router.py`
- Add conditional branch/node(s) only if current flow is insufficient.

6. Tests
- Add new unit tests under `tests/` for:
  - task classification behavior
  - action generation schema compliance
  - verification pass/retry behavior
  - submit payload shape

## 🛠️ Troubleshooting

### 1) LLM Context Overflow
Symptoms:
- Errors like `maximum context length`, `context_length_exceeded`, `prompt is too long`.

What the code already does:
- `clients/llm_client.py` trims old messages while preserving system + latest user context.
- Retries with exponential backoff and jitter.

What to check:
1. Reduce oversized prompts or context volume upstream.
2. Confirm `LLM_MAX_RETRIES`, `LLM_MAX_OUTPUT_TOKENS`, `VERIFICATION_MAX_OUTPUT_TOKENS` in `.env`.
3. Re-run `python -m unittest tests.test_llm_client_overflow`.

### 2) OpenAI Vision OCR Failure
Symptoms:
- Parser warning that image OCR failed or returned empty content.

What the code already does:
- In `tools/document_parser.py`, image resources are normalized and sent directly to OpenAI vision OCR.
- Parser keeps running even if OCR fails for a single image.

What to check:
1. Verify `OPENAI_API_KEY` in `.env`.
2. Confirm model configuration (`MODEL_NAME`) supports image input.
3. Check network access and API quota/rate limits.

### 3) Token Expiry or API Rate Limit
Symptoms:
- HTTP `401` (unauthorized) or `429` (rate limit).

What the code already does:
- `clients/competition_client.py` retries with backoff for retryable statuses.
- On `401`, client refreshes session and retries request.
- Session checkpoint is persisted under `storage/session_checkpoint.json`.

What to check:
1. Confirm `COMPETITION_BASE_URL` and `API_KEY` are valid.
2. Tune `HTTP_MAX_RETRIES`, `HTTP_BACKOFF_SECONDS`, `HTTP_TIMEOUT_SECONDS` for unstable networks.
3. Check `storage/agent.log` for endpoint-level retries and auth refresh events.

## 🔒 Security Notes
- Never commit real API keys or tokens.
- Keep `.env` local only.
- Keep logs and storage artifacts sanitized before sharing.
