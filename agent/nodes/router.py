# agent/nodes/router.py
from agent.state import InnerState, OuterState
from config import config
from core.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = config.MAX_RETRIES
MIN_CONFIDENCE = config.VERIFIER_MIN_CONFIDENCE

def route_rag_or_context(state: InnerState) -> str:
    """Điều hướng sang nhánh RAG hoặc context manager trong inner loop.

    Args:
        state: Trạng thái inner loop chứa cờ use_rag.

    Returns:
        str: "use_rag" khi bật RAG, ngược lại "use_context_manager".
    """
    if state.get("use_rag"):
        return "use_rag"
    return "use_context_manager"

def check_verification(state: InnerState) -> str:
    """Quyết định thoát hay lặp lại vòng self-correction.

    Args:
        state: Trạng thái inner loop chứa attempts, is_verified và confidence_score.

    Returns:
        str: "pass" nếu đạt điều kiện dừng hoặc quá số lần retry, ngược lại "retry".
    """
    if state.get("is_verified") and state.get("confidence_score", 0.0) >= MIN_CONFIDENCE:
        logger.info("[Verifiability] Passed with confidence=%.3f", state.get("confidence_score", 0.0))
        return "pass"

    if state.get("fallback_due_to_grounding"):
        logger.warning("[Verifiability] Grounding fallback activated; submitting conservative answer")
        return "pass"

    if state["attempts"] >= MAX_RETRIES:
        logger.warning("[Verifiability] Max retries reached; accepting current answer")
        return "pass" # Thoát vòng lặp
        
    logger.warning(
        "[Verifiability] Failed confidence=%.3f attempt=%s; retrying",
        state.get("confidence_score", 0.0),
        state.get("attempts"),
    )
    return "retry"


def route_after_action(state: InnerState) -> str:
    """Route after action generation based on whether a tool call is needed.

    Args:
        state: Trạng thái inner loop sau action_generation.

    Returns:
        str: "call_vision_tool" nếu có tool_calls, ngược lại "verifiability".
    """
    if state.get("tool_calls"):
        return "call_vision_tool"
    return "verifiability"

def route_outer_loop(state: OuterState) -> str:
    """Điều hướng vòng lặp ngoài sang xử lý task hoặc kết thúc.

    Args:
        state: Trạng thái outer loop chứa current_task và should_continue.

    Returns:
        str: "process_task" khi còn task cần xử lý, ngược lại "end".
    """
    if state.get("current_task") is None or not state.get("should_continue", True):
        return "end"
    return "process_task"