# agent/nodes/router.py
from devday_agent.agent.state import InnerState, OuterState
from config import config
from devday_agent.core.logger import get_logger

logger = get_logger(__name__)

MAX_RETRIES = config.MAX_RETRIES
MIN_CONFIDENCE = config.VERIFIER_MIN_CONFIDENCE

def route_rag_or_context(state: InnerState) -> str:
    """Route inner loop to RAG or full-context branch.

    Args:
        state: Inner-loop state containing the use_rag decision flag.

    Returns:
        str: "use_rag" when RAG is enabled, otherwise "use_context_manager".
    """
    if state.get("use_rag"):
        return "use_rag"
    return "use_context_manager"

def check_verification(state: InnerState) -> str:
    """Decide whether to stop or continue the self-correction loop.

    Args:
        state: Inner-loop state with attempts, verification status, and confidence.

    Returns:
        str: "pass" when stop criteria are met, otherwise "retry".
    """
    if state.get("is_verified") and state.get("confidence_score", 0.0) >= MIN_CONFIDENCE:
        logger.info("[Verifiability] Passed with confidence=%.3f", state.get("confidence_score", 0.0))
        return "pass"

    if state.get("fallback_due_to_grounding"):
        logger.warning("[Verifiability] Grounding fallback activated; submitting conservative answer")
        return "pass"

    if state["attempts"] >= MAX_RETRIES:
        logger.warning("[Verifiability] Max retries reached; accepting current answer")
        return "pass"  # Exit the retry loop.
        
    logger.warning(
        "[Verifiability] Failed confidence=%.3f attempt=%s; retrying",
        state.get("confidence_score", 0.0),
        state.get("attempts"),
    )
    return "retry"


def route_after_action(state: InnerState) -> str:
    """Route after action generation based on whether a tool call is needed.

    Args:
        state: Inner-loop state after action_generation.

    Returns:
        str: "call_vision_tool" when tool_calls is non-empty, otherwise "verifiability".
    """
    if state.get("tool_calls"):
        return "call_vision_tool"
    return "verifiability"

def route_outer_loop(state: OuterState) -> str:
    """Route the outer loop to task processing or termination.

    Args:
        state: Outer-loop state with current_task and should_continue.

    Returns:
        str: "process_task" when a task is available, otherwise "end".
    """
    if state.get("current_task") is None or not state.get("should_continue", True):
        return "end"
    return "process_task"