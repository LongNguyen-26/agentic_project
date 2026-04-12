# AI Agent (LangGraph)

## 🚀 Overview
This project is an AI Agent for the **AI Agent Challenge**.

It automates two task families from the competition server:
- **question-answering**: extract grounded answers from as-built project documents.
- **folder-organisation**: assign documents into valid target folders.

The runtime uses a nested LangGraph design:
- **Outer Loop** handles competition lifecycle (`auth -> fetch -> planning -> process_task -> submit`).
- **Inner Loop** handles task solving (`observability -> retrieval/context -> action_generation -> verifiability`) with automatic self-correction.

## 🧩 Prerequisites
- Python **>= 3.12**
- Network access to competition API server
- OpenAI API key for LLM/embedding operations
- Optional: local Ollama for Tier-2 vision OCR fallback

Required environment variables:
- `COMPETITION_BASE_URL`
- `API_KEY`
- `OPENAI_API_KEY`

Optional local vision setup:
- `OLLAMA_BASE_URL` (example: `http://localhost:11434`)
- `LOCAL_VISION_MODEL` (default: `Qwen/Qwen2.5-VL-3B-Instruct`)

If Ollama is not reachable, the parser automatically degrades to Tier-3 vision OCR (OpenAI).

## 📦 Installation
From the directory:

```bash
# 1) Create and activate your virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2) Install core dependencies
pip install -e .

# 3) (Optional) Install local OCR extras
pip install -e ".[ocr]"
```

Create your environment file:

```bash
copy .env.example .env
```

Then fill required keys in `.env`.

## Submission Setup & Execution Instructions
This section is intended for judge-side reproducible setup and execution.

### 1) Prerequisites
- Python **>= 3.12**
- `uv` package manager

### 2) Environment Setup
```bash
cp .env.example .env
```

Fill required keys in `.env`:
- `COMPETITION_BASE_URL`
- `API_KEY`
- `OPENAI_API_KEY`

On Windows `cmd`, you can use:

```bash
copy .env.example .env
```

### 3) Install Dependencies
Preferred (locked and reproducible from `uv.lock`):

```bash
uv sync
```

Fallback (if `uv` is unavailable):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### 4) Execute
```bash
python main.py
```

### 5) Run Tests
```bash
python -m unittest discover -s tests
```

## ⚡ Quick Start
Run the agent:

```bash
python main.py
```

Expected success signals in logs:
- `[agent] Starting VPP AI Agent runtime`
- `[agent] LangGraph compiled and execution loop starting`
- `[auth] ...` session/auth is ready
- `[task] Fetching next task`
- `[Graph] Node 'process_task' ...` inner loop produced result
- `[submit] Submit success for task_id=...`

When no task is available:
- `[loop] No more tasks available; stopping execution`

## 🗂️ Project Structure
- `agent/`: graph wiring, routing, and node logic (outer + inner loops)
- `clients/`: API client and LLM client wrappers
- `tools/`: document parser, context manager, RAG engine
- `models/`: typed schemas for API payloads and structured LLM outputs
- `core/`: logger, checkpoint, and shared exceptions
- `tests/`: unit tests for parser, loops, planning, overflow handling
- `storage/`: runtime checkpoints, cache, and logs

## 🧪 Testing
Run all tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Run a specific suite:

```bash
python -m unittest tests.test_llm_client_overflow
```

## 📚 More Docs
- `ARCHITECTURE.md`: detailed system map and design decisions
- `CONTRIBUTING.md`: developer workflow, extension guide, troubleshooting
