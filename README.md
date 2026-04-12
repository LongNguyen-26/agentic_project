# DUT_Trust AI Agent

## Overview
This repository contains a LangGraph-based competition agent for two task families:
- question-answering
- folder-organisation

The system uses a nested graph:
- Outer Loop: `auth -> fetch -> planning -> process_task -> submit`
- Inner Loop: `observability -> (setup_rag | setup_context) -> action_generation -> verifiability`

## Setup Instructions (For Organizers/Judges)

### 1. Prerequisites
- Python 3.12+
- Network access to the competition API
- OpenAI API key
- Recommended: uv package manager (lockfile-first workflow)

### 2. Prepare Environment Variables
Create `.env` from template:

Windows (cmd):
```bat
copy .env.example .env
```

Windows (PowerShell):
```powershell
Copy-Item .env.example .env
```

Linux/macOS:
```bash
cp .env.example .env
```

Set required keys in `.env`:
- `COMPETITION_BASE_URL`
- `API_KEY`
- `OPENAI_API_KEY`

### 3. Install Dependencies

#### Option A: uv sync from lockfile (Primary and Recommended)
This repository includes `uv.lock`, so this is the preferred installation path.

If uv is not installed yet:

Windows (PowerShell):
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Linux/macOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install runtime dependencies exactly from lockfile:

Windows/Linux/macOS:
```bash
uv sync --frozen
```

If you need to run tests, install dev dependencies as well:

```bash
uv sync --frozen --extra dev
```

#### Option B: venv + pip (Fallback only if uv is unavailable)

Windows (cmd):
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Linux/macOS:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

### 4. Validate Installation
Run a quick import check:

Preferred (uv-managed env):
```bash
uv run python -c "import langgraph, openai, instructor, pymupdf4llm, devday_agent; print('dependencies-ok')"
```

Alternative (if `.venv` is already activated):
```bash
python -c "import langgraph, openai, instructor, pymupdf4llm, devday_agent; print('dependencies-ok')"
```

## Execution Instructions

### 1. Run the agent

Preferred (uses uv-managed environment without manual activation):
```bash
uv run python -m devday_agent.main
```

Alternative (if you manually activated `.venv`):
```bash
python -m devday_agent.main
```

### 2. Expected runtime signals
- `[agent] Starting VPP AI Agent runtime`
- `[auth] Checking authentication state`
- `[task] Fetching next task`
- `[submit] Submit success for task_id=...`

### 3. Log location
- Main log file: `storage/agent.log`
- Session checkpoint: `storage/session_checkpoint.json`

## Testing Instructions

Preferred (after `uv sync --frozen --extra dev`):
```bash
uv run pytest src/tests -q
```

Alternative (if `.venv` already activated):
```bash
python -m pytest src/tests -q
```

Run parser-only tests:
```bash
uv run pytest src/tests/test_document_parser.py -q
```

## Reproducibility Notes
- Use `uv sync --frozen` to match the lockfile exactly.
- Use `uv sync --frozen --extra dev` when running tests.
- Use the same Python major/minor version across environments.
- Keep `.env` local and never commit secrets.

## Repository Structure
- `src/devday_agent/agent/`: graph composition, routers, and nodes
- `src/devday_agent/clients/`: competition and LLM clients
- `src/devday_agent/tools/`: parser, context manager, RAG, vision tool
- `src/devday_agent/models/`: pydantic schemas for API and LLM structured outputs
- `src/devday_agent/core/`: logger, checkpoints, shared exceptions
- `src/tests/`: unit tests

## Additional Documents
- `ARCHITECTURE.md`
- `CONTRIBUTING.md`
