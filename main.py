# main.py
import sys
import time

from core.logger import get_logger, setup_logging
from core.checkpoint import load_checkpoint


logger = get_logger(__name__)

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
        
        for output in agent_app.stream(initial_state, config=config):
            # In ra các node đang được thực thi
            for node_name, state_update in output.items():
                logger.debug("[loop] Node completed: %s", node_name.upper())

                # Một số node có thể không ghi state (None), cần chặn an toàn.
                if isinstance(state_update, dict) and state_update.get("should_continue") is False:
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