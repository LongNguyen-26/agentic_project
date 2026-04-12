# agent/graph.py
"""Define the full VPP agent StateGraph with Outer/Inner loop topology.

High-level flow:

        OUTER LOOP (Competition Lifecycle)
        +------+     +-------+     +----------+     +-------------+     +--------+
        | auth | --> | fetch | --> | planning | --> | process_task| --> | submit |
        +------+     +-------+     +----------+     +-------------+     +--------+
                                         |                                                    |
                                         +------------------- no task ------------------------+
                                                                                 -> END

        INNER LOOP (Per-task Processing, invoked from process_task)
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

from typing import Any

from langgraph.graph import StateGraph, END

from devday_agent.core.logger import get_logger
from devday_agent.agent.state import InnerState, OuterState

# Import node implementations from devday_agent.agent/nodes.
from devday_agent.agent.nodes.inner_loop import (
    observability_node, 
    setup_rag_node, 
    setup_context_manager_node, 
    action_generation_node, 
    verifiability_node,
    vision_tool_node,
)
from devday_agent.agent.nodes.outer_loop import auth_node, fetch_task_node, planning_node, submit_node
from devday_agent.agent.nodes.router import route_rag_or_context, check_verification, route_after_action, route_outer_loop


logger = get_logger(__name__)

# ==========================================
# 1. BUILD INNER GRAPH (Task Processing)
# ==========================================
inner_workflow = StateGraph(InnerState)

# Add nodes.
inner_workflow.add_node("observability", observability_node)
inner_workflow.add_node("setup_rag", setup_rag_node)
inner_workflow.add_node("setup_context", setup_context_manager_node)
inner_workflow.add_node("action_generation", action_generation_node)
inner_workflow.add_node("vision_tool", vision_tool_node)
inner_workflow.add_node("verifiability", verifiability_node)

# Define edges.
inner_workflow.set_entry_point("observability")

# From observability, route to RAG or full-context branch.
inner_workflow.add_conditional_edges(
    "observability",
    route_rag_or_context,
    {
        "use_rag": "setup_rag",
        "use_context_manager": "setup_context"
    }
)

# Both branches return to action generation.
inner_workflow.add_edge("setup_rag", "action_generation")
inner_workflow.add_edge("setup_context", "action_generation")

# Route action generation to vision tool or verifier.
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

# Self-correction loop (verifiability -> action_generation).
inner_workflow.add_conditional_edges(
    "verifiability",
    check_verification,
    {
        "retry": "action_generation",  # Verifier feedback guides the next action step.
        "pass": END
    }
)

# Compile inner graph.
inner_app = inner_workflow.compile()

# ==========================================
# 2. WRAP INNER GRAPH AS AN OUTER NODE
# ==========================================
def process_task_node(state: OuterState) -> dict[str, Any]:
    """Bridge the outer loop to execute inner loop for one task.

    Args:
        state: Outer-loop state with current_task, auth info, and planning hints.

    Returns:
        dict[str, Any]: task_result assembled from final inner-loop draft_answer.
    """
    task = state["current_task"]
    logger.info("[task] Processing task_id=%s type=%s", task.get("id"), task.get("type"))

    # Initialize inner-loop state.
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
        "answer_log": "",
        "used_tools": [],
        "attempts": 0,
        "fallback_due_to_grounding": False,
        "is_verified": False,
    }

    # Execute inner graph.
    final_inner_state = inner_app.invoke(initial_inner_state)

    # Extract final verified answer for outer loop submission.
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

# Add nodes.
outer_workflow.add_node("auth", auth_node)
outer_workflow.add_node("fetch", fetch_task_node)
outer_workflow.add_node("planning", planning_node)
outer_workflow.add_node("process_task", process_task_node)  # Node that executes inner graph.
outer_workflow.add_node("submit", submit_node)

# Define edges.
outer_workflow.set_entry_point("auth")
outer_workflow.add_edge("auth", "fetch")

# If a task is available, continue processing; otherwise end.
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
outer_workflow.add_edge("submit", "fetch")  # Fetch next task right after submit.

# Compile full system graph.
agent_app = outer_workflow.compile()