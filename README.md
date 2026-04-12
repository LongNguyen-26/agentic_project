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
- Optional but recommended: `uv` package manager for reproducible installs

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

#### Option A: uv (Recommended)
Use lock-based installation for reproducibility.

Windows/Linux/macOS:
```bash
uv sync --frozen
```

#### Option B: venv + pip (Fallback)

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

```bash
python -c "import langgraph, openai, instructor, pymupdf4llm; print('dependencies-ok')"
```

## Execution Instructions

### 1. Run the agent
```bash
python main.py
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

Run full test suite:
```bash
python -m pytest tests -q
```

Run parser-only tests:
```bash
python -m pytest tests/test_document_parser.py -q
```

## Reproducibility Notes
- Use `uv sync --frozen` to match the lockfile exactly.
- Use the same Python major/minor version across environments.
- Keep `.env` local and never commit secrets.

## Repository Structure
- `agent/`: graph composition, routers, and nodes
- `clients/`: competition and LLM clients
- `tools/`: parser, context manager, RAG, vision tool
- `models/`: pydantic schemas for API and LLM structured outputs
- `core/`: logger, checkpoints, shared exceptions
- `tests/`: unit tests

## Additional Documents
- `ARCHITECTURE.md`
- `CONTRIBUTING.md`
