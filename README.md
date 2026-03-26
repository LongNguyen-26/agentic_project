# VPP AI Agent Competition — Example Agent

A LangGraph-based agent that completes the competition loop end-to-end.

## Architecture

```
START
  └─ authenticate          POST /sessions → get Bearer token
       └─ fetch_task       GET /tasks/next → receive task (404 = done)
            └─ process_task  reads files, reasons, produces answers
                 └─ submit_answer  POST /submissions
                      └─ fetch_task  (loops back until 404)
```

The **outer graph** (`StateGraph`) manages the competition lifecycle (authenticate → fetch → process → submit → repeat).

The **inner logic** handles each task via two specialised paths:

| Task type | Flow |
|---|---|
| Question Answering | `ContextManager` accumulates file content → `generate_summary()` → `QAHandler` extracts a clean final answer |
| Folder Organisation | `SortHandler` classifies each file → `save_file` saves locally → `generate_summary()` for the thought log |

### Key components

| Component | Role |
|---|---|
| `ContextManager` | Manages conversation history and summary generation; handles context-overflow by auto-trimming and retrying |
| `QAHandler` | Detects answer type and extracts a verified final answer using a reasoning model |
| `SortHandler` | Classifies documents into the correct folder using an LLM |

### Tools available to the inner agent

| Tool | Signature | When used |
|---|---|---|
| `read_resource` | `read_resource(file_path)` | Every task — downloads and reads a file (PDF → Markdown text, image → base64) |
| `save_file` | `save_file(file_path, folder_name)` | Folder Organisation tasks only — saves the file to the local organised folder |

## Setup

```bash
cp .env.example .env   # fill in COMPETITION_BASE_URL, API_KEY, OPENAI_API_KEY
uv sync
uv run python main.py
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `COMPETITION_BASE_URL` | Yes | Base URL of the competition server |
| `API_KEY` | Yes | Your competition API key |
| `OPENAI_API_KEY` | Yes | OpenAI key (used for the main agent and QA extraction) |
| `MODEL_NAME` | No | Chat model for the main agent (default: `gpt-4o`) |
| `REASONING_MODEL` | No | Reasoning model for answer extraction (default: `o3`) |
| `VLM_MODEL_NAME` | No | Local VLM for scanned PDF OCR (default: `Qwen/Qwen2.5-VL-3B-Instruct`) |
| `DATA_DIR` | No | Local directory for Folder Organisation output (default: `organised_files`) |

## File layout

```
examples/
  main.py                    entry point — builds and streams the graph
  settings.py                loads .env and exposes validated config values
  agent/
    agent.py                 outer StateGraph + all node/edge logic
    context_manager.py       conversation history + summary generation
    qa_handler.py            answer type detection + final answer extraction
    sort_handler.py          folder classification for Folder Organisation tasks
    utils.py                 logging helpers
  tools/
    read_resource.py         download + parse files (PDF, images, Office docs)
    save_file.py             download + save to organised local folder
    _session.py              internal session/resource registry
  prompts/
    prompt_templates.py      all prompt-building functions
    sys_prompts.py           system prompt entry point
    user_prompts.py          per-task user message formatter
  .env.example
  pyproject.toml
```

## PDF handling

`read_resource` uses a three-tier pipeline for PDFs:

1. **pymupdf** — fast text extraction; works for text-layer PDFs.
2. **Qwen2.5-VL (local VLM)** — visual understanding for scanned or complex PDFs; runs on MPS/CUDA/CPU. The model (~6 GB) is downloaded from Hugging Face on first use.
3. **GPT-4o vision fallback** — used only when the VLM output is too short.

## Extending the agent

- **Better QA answers** — replace the `QAHandler` extraction step with structured output (`model.with_structured_output(...)`) for stricter answer parsing.
- **Retry logic** — wrap `process_task` with LangGraph's built-in retry support.
- **Checkpointing** — pass a `MemorySaver` to `build_graph()` to persist state across runs.
- **Streaming** — switch `graph.stream(stream_mode="updates")` to `"values"` to observe full state at each step.
- **Swap the VLM** — set `VLM_MODEL_NAME` to any Qwen2.5-VL variant (e.g. `Qwen/Qwen2.5-VL-7B-Instruct`) for higher OCR quality.
