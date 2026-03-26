# agent/nodes/outer_loop.py
from agent.state import OuterState
from clients.competition_client import APIClient
from core.checkpoint import _persist_session_checkpoint
from core.logger import get_logger

client = None
logger = get_logger(__name__)


def _get_client() -> APIClient:
    global client
    if client is None:
        client = APIClient()
    return client

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
    logger.info("[task] Fetching next task")
    task = _get_client().fetch_next_task()

    if not task:
        logger.info("[loop] No tasks returned by server")
        return {"current_task": None, "should_continue": False}

    task_type = task.get("type") or "question-answering"
    return {"current_task": {**task, "type": task_type}, "should_continue": True}

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
    else:
        logger.warning("[submit] Submit failed for task_id=%s", task_id)

    return {"current_task": None, "task_result": None}