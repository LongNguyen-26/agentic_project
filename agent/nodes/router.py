# agent/nodes/router.py
from agent.state import InnerState, OuterState
from config import config
from core.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = config.MAX_RETRIES
MIN_CONFIDENCE = config.VERIFIER_MIN_CONFIDENCE

def route_rag_or_context(state: InnerState) -> str:
    """Dùng RAG cho QA khi số tài liệu vượt ngưỡng cấu hình."""
    if state.get("use_rag"):
        return "use_rag"
    return "use_context_manager"

def check_verification(state: InnerState) -> str:
    """
    Nút chặn Self-Correction: Kiểm tra xem đáp án đã đạt chuẩn chưa.
    Nếu confidence < ngưỡng hoặc is_verified = False -> Quay lại action_generation
    """
    if state["attempts"] >= MAX_RETRIES:
        logger.warning("[Verifiability] Max retries reached; accepting current answer")
        return "pass" # Thoát vòng lặp
        
    if state.get("is_verified") and state.get("confidence_score", 0.0) >= MIN_CONFIDENCE:
        logger.info("[Verifiability] Passed with confidence=%.3f", state.get("confidence_score", 0.0))
        return "pass"
        
    logger.warning(
        "[Verifiability] Failed confidence=%.3f attempt=%s; retrying",
        state.get("confidence_score", 0.0),
        state.get("attempts"),
    )
    return "retry"

def route_outer_loop(state: OuterState) -> str:
    """Kiểm tra xem BTC còn task không, nếu không thì kết thúc."""
    if state.get("current_task") is None or not state.get("should_continue", True):
        return "end"
    return "process_task"