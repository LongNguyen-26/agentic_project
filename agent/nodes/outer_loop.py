# agent/nodes/outer_loop.py
from agent.state import OuterState
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

def auth_node(state: OuterState) -> dict:
    """Xử lý xác thực và phục hồi phiên."""
    logger.info("[auth] Checking authentication state")
    local_client = _get_client()

    if not state.get("access_token"):
        success = local_client.authenticate()
        if success:
            _persist_session_checkpoint(local_client.session_id or "", local_client.access_token or "")
            return {"session_id": local_client.session_id, "access_token": local_client.access_token}
        return {"should_continue": False, "error": "Authentication failed"}

    local_client.session_id = state["session_id"]
    local_client.access_token = state["access_token"]
    return {}

def fetch_task_node(state: OuterState) -> dict:
    """Kéo task mới từ server."""
    # NẾU TASK CŨ CHƯA NỘP THÀNH CÔNG, BỎ QUA VIỆC FETCH VÀ RETRY TASK HIỆN TẠI
    if state.get("current_task") is not None:
        logger.info("[task] Retrying existing task_id=%s", state["current_task"]["id"])
        return {"should_continue": True}
    
    logger.info("[task] Fetching next task")
    task = _get_client().fetch_next_task()

    if not task:
        logger.info("[loop] No tasks returned by server")
        return {"current_task": None, "planning_hints": "", "should_continue": False}

    task_type = task.get("type") or "question-answering"
    prompt_template = task.get("prompt_template", "")
    
    # ---> THÊM DÒNG LOG NÀY <---
    short_prompt = (prompt_template[:100] + "...") if len(prompt_template) > 100 else prompt_template
    logger.info("[task] Received task_id=%s, task_type=%s, prompt='%s'", task.get("id"), task_type, short_prompt)
    return {"current_task": {**task, "type": task_type}, "should_continue": True}


def planning_node(state: OuterState) -> dict:
    """Extract high-level hints/cautions before invoking the inner graph."""
    task = state.get("current_task")
    if not task:
        return {"planning_hints": ""}

    prompt_template = task.get("prompt_template", "")
    if not prompt_template.strip():
        return {"planning_hints": ""}

    logger.info("[planning] Extracting pre-execution hints for task_id=%s", task.get("id"))
    hints = _get_llm_service().extract_planning_hints(prompt_template)
    return {"planning_hints": hints}

def submit_node(state: OuterState) -> dict:
    """Nộp kết quả bài làm lên server."""
    task_id = state["current_task"]["id"]
    result = state.get("task_result")

    if result is None:
        logger.warning("[submit] Missing result for task_id=%s; skip submit", task_id)
        return {}

    logger.info("[submit] Submitting task_id=%s", task_id)
    success = _get_client().submit_task_result(task_id, result)

    if success:
        logger.info("[submit] Submit success for task_id=%s", task_id)
        # CHỈ XÓA STATE KHI THÀNH CÔNG
        return {"current_task": None, "planning_hints": "", "task_result": None}
    else:
        logger.warning("[submit] Submit failed for task_id=%s, keeping task in state to retry", task_id)
        # NẾU LỖI, GIỮ NGUYÊN STATE ĐỂ RETRY
        return {}