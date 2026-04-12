# main.py
import sys
import time
from typing import Any

from devday_agent.core.logger import get_logger, setup_logging
from devday_agent.core.checkpoint import load_checkpoint


logger = get_logger(__name__)


def _is_meaningful_value(value: Any) -> bool:
    """Return True when value contains useful non-empty payload."""
    return value not in (None, "", [], {})


def _extract_important_updates(state_update: dict) -> dict:
    important = {}

    for key in ("task_result", "answers", "error", "confidence", "confidence_score", "should_continue"):
        if key in state_update and _is_meaningful_value(state_update.get(key)):
            important[key] = state_update.get(key)

    # Extract compact task details for concise log output.
    current_task = state_update.get("current_task")
    if isinstance(current_task, dict):
        if _is_meaningful_value(current_task.get("type")):
            important["task_type"] = current_task.get("type")
        if _is_meaningful_value(current_task.get("prompt_template")):
            pt = current_task.get("prompt_template")
            # Truncate prompt template to avoid oversized log lines.
            important["prompt_template"] = (pt[:100] + "...") if len(pt) > 100 else pt

    task_result = state_update.get("task_result")
    if isinstance(task_result, dict):
        if _is_meaningful_value(task_result.get("answers")):
            important["answers"] = task_result.get("answers")
        if _is_meaningful_value(task_result.get("confidence")):
            important["confidence"] = task_result.get("confidence")

    return important

def main() -> None:
    setup_logging()
    logger.info("[agent] Starting VPP AI Agent runtime")
    from devday_agent.agent.graph import agent_app  # Import after logging bootstrap.
    
    # 1. Restore session checkpoint from disk when available.
    session_id, access_token = load_checkpoint()
    
    # 2. Initialize outer-loop state.
    initial_state = {
        "session_id": session_id,
        "access_token": access_token,
        "current_task": None,
        "planning_hints": "",
        "task_result": None,
        "error": None,
        "should_continue": True
    }
    
    logger.info("[agent] LangGraph compiled and execution loop starting")
    
    # 3. Run the graph loop continuously.
    # Use .stream() instead of .invoke() for step-by-step logging.
    try:
        # The submit -> fetch edge forms a persistent loop.
        # Keep recursion_limit high for long-running task processing.
        config = {"recursion_limit": 1000} 
        
        for output in agent_app.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, state_update in output.items():
                if not isinstance(state_update, dict):
                    logger.info("[Graph] Node '%s' completed", node_name)
                    continue

                important_updates = _extract_important_updates(state_update)
                if important_updates:
                    updates_text = ", ".join(
                        f"{key}: {value}" for key, value in important_updates.items()
                    )
                    logger.info("[Graph] Node '%s' updated -> %s", node_name, updates_text)
                else:
                    logger.debug("[Graph] Node '%s' completed (no important updates)", node_name)

                # Some nodes may emit no state updates; guard safely.
                if state_update.get("should_continue") is False:
                    logger.info("[loop] No more tasks available; stopping execution")
                    return
                    
            # Adaptive throttling between node executions.
            if node_name in ["fetch", "submit"]:
                time.sleep(0.5)  # Slightly longer pause after API interactions.
            else:
                time.sleep(0.05)  # Short pause for local compute nodes.
            
    except KeyboardInterrupt:
        logger.warning("[agent] Process interrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        logger.error("[agent] Unexpected fatal error: %s", e, exc_info=True)
        # Optional: plug in external alerting here (Telegram/Discord, etc.).
        sys.exit(1)

if __name__ == "__main__":
    main()