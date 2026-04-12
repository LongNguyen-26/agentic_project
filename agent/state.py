# agent/state.py
from typing import Any, Dict, List, Optional, TypedDict

class InnerState(TypedDict):
    """Trạng thái cho một task cụ thể (vòng lặp trong)."""
    task_id: str
    task_type: str
    prompt_template: str
    session_id: Optional[str]
    access_token: Optional[str]
    resources: List[Dict[str, str]]
    parsed_documents: List[Dict[str, Any]]
    parsed_text: str
    planning_hints: str

    use_rag: bool
    retrieved_context: str

    action_plan: Dict[str, Any]
    draft_answer: Dict[str, Any]
    tool_calls: List[str]
    vision_prompt: str
    tool_observations: List[str]

    confidence_score: float
    is_verified: bool
    verification_feedback: str
    answer_log: str
    attempts: int
    fallback_due_to_grounding: bool
    used_tools: List[str]

class OuterState(TypedDict):
    """Trạng thái tổng thể của cuộc thi (vòng lặp ngoài)."""
    session_id: Optional[str]
    access_token: Optional[str]

    current_task: Optional[Dict[str, Any]]
    planning_hints: str
    task_result: Optional[Dict[str, Any]]

    error: Optional[str]
    should_continue: bool