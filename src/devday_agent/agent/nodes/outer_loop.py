# agent/nodes/outer_loop.py
from typing import Any

from agent.state import OuterState
from agent.prompts.sys_prompts import SYS_CLASSIFY_TASK, SYS_PLANNING_HINTS
from agent.prompts.user_prompt import build_planning_hints_prompt, build_task_classification_prompt
from clients.competition_client import APIClient
from clients.llm_client import LLMService
from core.checkpoint import _persist_session_checkpoint
from core.logger import get_logger

client = None
llm_service = None
logger = get_logger(__name__)


def _get_client() -> APIClient:
    global client
    if client is None:
        client = APIClient()
    return client


def _get_llm_service() -> LLMService:
    """Lazily initialize shared LLM service for outer-loop planning."""
    global llm_service
    if llm_service is None:
        llm_service = LLMService()
    return llm_service

def auth_node(state: OuterState) -> dict[str, Any]:
    """Authenticate the session and sync credentials into the API client.

    Args:
        state: Outer-loop state that may contain existing session credentials.

    Returns:
        dict[str, Any]: Updated credentials or a stop flag when authentication fails.
    """
    logger.info("[auth] Checking authentication state")
    local_client = _get_client()

    if not state.get("access_token"):
        success = local_client.authenticate()
        if success:
            _persist_session_checkpoint(local_client.session_id or "", local_client.access_token or "")
            return {"session_id": local_client.session_id, "access_token": local_client.access_token}
        return {"should_continue": False, "error": "Authentication failed"}

    local_client.session_id = state.get("session_id")
    local_client.access_token = state.get("access_token")
    return {}

def fetch_task_node(state: OuterState) -> dict[str, Any]:
    """Fetch the next task and infer task_type if the API omitted it.

    Args:
        state: Outer-loop state with current task and loop-control fields.

    Returns:
        dict[str, Any]: Normalized current_task plus should_continue routing flag.
    """
    # If the previous task was not submitted successfully, retry it instead of fetching.
    existing_task = state.get("current_task")
    if existing_task is not None:
        logger.info("[task] Retrying existing task_id=%s", existing_task.get("id"))
        return {"should_continue": True}
    
    logger.info("[task] Fetching next task")
    task = _get_client().fetch_next_task()

    if not task:
        logger.info("[loop] No tasks returned by server")
        return {"current_task": None, "planning_hints": "", "should_continue": False}

    # Prefer API-provided task_type; fallback to local rules and then LLM classification.
    task_type = task.get("type")
    prompt_template = task.get("prompt_template", "")
    
    if not task_type:
        # Step 1: fast rule-based classification.
        prompt_lower = prompt_template.lower()
        folder_keywords = [
            "フォルダへ配置", "フォルダに分類", "仕分け",  # Japanese
            "folder", "sort", "organize", "organise", "classify",  # English
        ]
        # QA-oriented keywords used as fast hints.
        qa_keywords = [
            "特定し", "確認せよ", "取扱説明書", "注意事項",  # Japanese
            "question", "answer", "extract", "identify", "confirm",  # English
        ]

        if any(keyword in prompt_lower for keyword in folder_keywords):
            logger.info("[task] Fast classification: 'folder-organisation' (rule-based matched)")
            task_type = "folder-organisation"
        elif any(keyword in prompt_lower for keyword in qa_keywords):
            logger.info("[task] Fast classification: 'question-answering' (rule-based matched)")
            task_type = "question-answering"
        # Step 2: fallback to LLM when no rule matches.
        else:
            logger.info("[task] Missing task type from API, classifying via LLM...")
            task_type = _get_llm_service().classify_task_type(
                system_prompt=SYS_CLASSIFY_TASK,
                user_prompt=build_task_classification_prompt(prompt_template),
            )
    
    # Keep prompt preview short to avoid noisy logs.
    short_prompt = (prompt_template[:100] + "...") if len(prompt_template) > 100 else prompt_template
    logger.info("[task] Received task_id=%s, task_type=%s, prompt='%s'", task.get("id"), task_type, short_prompt)
    return {"current_task": {**task, "type": task_type}, "should_continue": True}


def planning_node(state: OuterState) -> dict[str, Any]:
    """Extract planning hints from prompt instructions before inner execution.

    Args:
        state: Outer-loop state carrying the current task prompt.

    Returns:
        dict[str, Any]: planning_hints for downstream action generation.
    """
    task = state.get("current_task")
    if not task:
        return {"planning_hints": ""}

    prompt_template = task.get("prompt_template", "")
    if not prompt_template.strip():
        return {"planning_hints": ""}

    logger.info("[planning] Extracting pre-execution hints for task_id=%s", task.get("id"))
    hints = _get_llm_service().extract_planning_hints(
        system_prompt=SYS_PLANNING_HINTS,
        user_prompt=build_planning_hints_prompt(prompt_template),
    )
    return {"planning_hints": hints}

def submit_node(state: OuterState) -> dict[str, Any]:
    """Submit task result and clear state only after successful submission.

    Args:
        state: Outer-loop state with current task and produced task result.

    Returns:
        dict[str, Any]: Updated state payload after submit attempt.
    """
    task = state.get("current_task")
    if not task:
        logger.warning("[submit] Missing current_task; skip submit")
        return {}

    task_id = task.get("id")
    result = state.get("task_result")

    if result is None:
        logger.warning("[submit] Missing result for task_id=%s; skip submit", task_id)
        return {}

    logger.info("[submit] Submitting task_id=%s", task_id)
    success = _get_client().submit_task_result(task_id, result)

    if success:
        logger.info("[submit] Submit success for task_id=%s", task_id)
        # Only clear task state after a successful submission.
        return {"current_task": None, "planning_hints": "", "task_result": None}

    logger.warning("[submit] Submit failed for task_id=%s, keeping task in state to retry", task_id)
    # Keep current state unchanged so the task can be retried.
    return {}