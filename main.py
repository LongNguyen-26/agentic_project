# main.py
import sys
import time

from core.logger import get_logger, setup_logging
from core.checkpoint import load_checkpoint


logger = get_logger(__name__)


def _is_meaningful_value(value):
    return value not in (None, "", [], {})


def _extract_important_updates(state_update: dict) -> dict:
    important = {}

    for key in ("task_result", "answers", "error", "confidence", "confidence_score", "should_continue"):
        if key in state_update and _is_meaningful_value(state_update.get(key)):
            important[key] = state_update.get(key)

    # ---> THÊM LOGIC TRÍCH XUẤT THÔNG TIN TASK TẠI ĐÂY <---
    current_task = state_update.get("current_task")
    if isinstance(current_task, dict):
        if _is_meaningful_value(current_task.get("type")):
            important["task_type"] = current_task.get("type")
        if _is_meaningful_value(current_task.get("prompt_template")):
            pt = current_task.get("prompt_template")
            # Cắt ngắn prompt template (ví dụ 100 ký tự) để log không bị quá dài
            important["prompt_template"] = (pt[:100] + "...") if len(pt) > 100 else pt

    task_result = state_update.get("task_result")
    if isinstance(task_result, dict):
        if _is_meaningful_value(task_result.get("answers")):
            important["answers"] = task_result.get("answers")
        if _is_meaningful_value(task_result.get("confidence")):
            important["confidence"] = task_result.get("confidence")

    return important

def main():
    setup_logging()
    logger.info("[agent] Starting VPP AI Agent runtime")
    from agent.graph import agent_app  # Import after logging bootstrap.
    
    # 1. Thử phục hồi Session từ ổ cứng (để không phải đăng nhập lại nếu bị crash)
    session_id, access_token = load_checkpoint()
    
    # 2. Khởi tạo Trạng thái Vòng lặp ngoài (Outer State)
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
    
    # 3. Chạy vòng lặp vô hạn của đồ thị
    # Sử dụng .stream() thay vì .invoke() để in ra log từng bước mượt mà hơn
    try:
        # Trong graph.py, luồng đi từ submit -> fetch tạo thành vòng lặp.
        # Chúng ta dùng "recursion_limit" cao để agent cắm chuột chạy liên tục.
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

                # Một số node có thể không ghi state (None), cần chặn an toàn.
                if state_update.get("should_continue") is False:
                    logger.info("[loop] No more tasks available; stopping execution")
                    return
                    
            time.sleep(1) # Nghỉ 1s giữa các node để tránh quá tải CPU/Rate limit
            
    except KeyboardInterrupt:
        logger.warning("[agent] Process interrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        logger.error("[agent] Unexpected fatal error: %s", e, exc_info=True)
        # Ở đây có thể thêm logic gửi thông báo qua Telegram/Discord cho dev
        sys.exit(1)

if __name__ == "__main__":
    main()