# agent/graph.py
"""Định nghĩa StateGraph tổng thể cho VPP agent theo mô hình Outer/Inner Loop.

Luồng tổng quan:

        OUTER LOOP (Competition Lifecycle)
        +------+     +-------+     +----------+     +-------------+     +--------+
        | auth | --> | fetch | --> | planning | --> | process_task| --> | submit |
        +------+     +-------+     +----------+     +-------------+     +--------+
                                         |                                                    |
                                         +------------------- no task ------------------------+
                                                                                 -> END

        INNER LOOP (Per-task Processing, gọi trong process_task)
        +---------------+
        | observability |
        +---------------+
                        |
                        v
            route_rag_or_context
                /             \\
             v               v
        +----------+   +--------------+
        | setup_rag|   | setup_context|
        +----------+   +--------------+
                 \\             /
                    v           v
                +-------------------+
                | action_generation |
                +-------------------+
                          |
                          v
                  route_after_action
                    |              |
                    v              v
              +-------------+  +---------------+
              | vision_tool |  | verifiability |
              +-------------+  +---------------+
                    |              |
                    +------>-------+
                          |
                          v
                   check_verification
                      |         |
                     pass     retry
                      |         |
                      v         +------> action_generation
                     END
"""

from langgraph.graph import StateGraph, END

from core.logger import get_logger
from .state import InnerState, OuterState

# Hướng dẫn: Import các hàm node của bạn (Tự định nghĩa logic bên trong file nodes/*)
from .nodes.inner_loop import (
    observability_node, 
    setup_rag_node, 
    setup_context_manager_node, 
    action_generation_node, 
    verifiability_node,
    vision_tool_node,
)
from .nodes.outer_loop import auth_node, fetch_task_node, planning_node, submit_node
from .nodes.router import route_rag_or_context, check_verification, route_after_action, route_outer_loop


logger = get_logger(__name__)

# ==========================================
# 1. BUILD INNER GRAPH (Task Processing)
# ==========================================
inner_workflow = StateGraph(InnerState)

# Thêm Nodes
inner_workflow.add_node("observability", observability_node)
inner_workflow.add_node("setup_rag", setup_rag_node)
inner_workflow.add_node("setup_context", setup_context_manager_node)
inner_workflow.add_node("action_generation", action_generation_node)
inner_workflow.add_node("vision_tool", vision_tool_node)
inner_workflow.add_node("verifiability", verifiability_node)

# Định nghĩa Luồng (Edges)
inner_workflow.set_entry_point("observability")

# Từ Observability -> Quyết định dùng RAG hay Context
inner_workflow.add_conditional_edges(
    "observability",
    route_rag_or_context,
    {
        "use_rag": "setup_rag",
        "use_context_manager": "setup_context"
    }
)

# Cả RAG và Context đều dẫn về Action Generation
inner_workflow.add_edge("setup_rag", "action_generation")
inner_workflow.add_edge("setup_context", "action_generation")

# Action Generation -> Vision Tool hoặc Verifiability
inner_workflow.add_conditional_edges(
    "action_generation",
    route_after_action,
    {
        "call_vision_tool": "vision_tool",
        "verifiability": "verifiability",
    }
)

# ReAct loop: Tool observation -> Action Generation
inner_workflow.add_edge("vision_tool", "action_generation")

# Vòng lặp Self-Correction (Verifiability -> Action Generation)
inner_workflow.add_conditional_edges(
    "verifiability",
    check_verification,
    {
        "retry": "action_generation", # Phản hồi từ verifiability sẽ giúp action_generation làm tốt hơn
        "pass": END
    }
)

# Đóng gói Inner Graph
inner_app = inner_workflow.compile()

# ==========================================
# 2. BỌC INNER GRAPH VÀO 1 NODE CỦA OUTER GRAPH
# ==========================================
def process_task_node(state: OuterState) -> dict:
    """Cầu nối outer loop để thực thi inner loop cho một task cụ thể.

    Args:
        state: Trạng thái outer loop chứa current_task, auth info và planning hints.

    Returns:
        dict: task_result đã được tổng hợp từ draft_answer cuối của inner loop.
    """
    task = state["current_task"]
    logger.info("[task] Processing task_id=%s type=%s", task.get("id"), task.get("type"))

    # Khởi tạo trạng thái cho vòng lặp trong
    initial_inner_state = {
        "task_id": task.get("id"),
        "task_type": task.get("type") or "question-answering",
        "prompt_template": task.get("prompt_template", ""),
        "planning_hints": state.get("planning_hints", ""),
        "session_id": state.get("session_id"),
        "access_token": state.get("access_token"),
        "resources": task.get("resources", []),
        "parsed_documents": [],
        "parsed_text": "",
        "use_rag": False,
        "retrieved_context": "",
        "action_plan": {},
        "draft_answer": {},
        "tool_calls": [],
        "vision_prompt": "",
        "tool_observations": [],
        "confidence_score": 0.0,
        "verification_feedback": "",
        "used_tools": [],
        "attempts": 0,
        "is_verified": False,
    }

    # KÍCH HOẠT INNER GRAPH
    final_inner_state = inner_app.invoke(initial_inner_state)

    # Lấy đáp án cuối cùng (đã qua kiểm duyệt) trả về cho vòng lặp ngoài
    draft = final_inner_state.get("draft_answer", {})
    task_result = {
        "answers": draft.get("answers", []),
        "thought_log": draft.get("thought_log", ""),
        "used_tools": draft.get("used_tools", final_inner_state.get("used_tools", [])),
    }
    return {
        "task_result": task_result
    }

# ==========================================
# 3. BUILD OUTER GRAPH (Competition Loop)
# ==========================================
outer_workflow = StateGraph(OuterState)

# Thêm Nodes
outer_workflow.add_node("auth", auth_node)
outer_workflow.add_node("fetch", fetch_task_node)
outer_workflow.add_node("planning", planning_node)
outer_workflow.add_node("process_task", process_task_node) # Node gọi Inner Graph
outer_workflow.add_node("submit", submit_node)

# Định nghĩa Luồng (Edges)
outer_workflow.set_entry_point("auth")
outer_workflow.add_edge("auth", "fetch")

# Nếu lấy được task -> process, nếu rỗng -> END (chờ task mới hoặc tắt agent)
outer_workflow.add_conditional_edges(
    "fetch",
    route_outer_loop,
    {
        "process_task": "planning",
        "end": END
    }
)

outer_workflow.add_edge("planning", "process_task")

outer_workflow.add_edge("process_task", "submit")
outer_workflow.add_edge("submit", "fetch") # Nộp xong lập tức lấy task mới

# Đóng gói toàn bộ hệ thống
agent_app = outer_workflow.compile()