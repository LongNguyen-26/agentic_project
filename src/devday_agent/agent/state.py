# agent/state.py
from typing import Any, NotRequired, Required, TypedDict

class InnerState(TypedDict):
    """State for processing a single task in the inner loop."""
    task_id: Required[str]
    task_type: Required[str]
    prompt_template: Required[str]
    resources: Required[list[dict[str, str]]]
    parsed_documents: Required[list[dict[str, Any]]]
    parsed_text: Required[str]

    use_rag: Required[bool]
    retrieved_context: Required[str]

    action_plan: Required[dict[str, Any]]
    draft_answer: Required[dict[str, Any]]
    tool_calls: Required[list[str]]
    vision_prompt: Required[str]
    tool_observations: Required[list[str]]

    confidence_score: Required[float]
    is_verified: Required[bool]
    verification_feedback: Required[str]
    answer_log: Required[str]
    attempts: Required[int]
    fallback_due_to_grounding: Required[bool]
    used_tools: Required[list[str]]

    session_id: NotRequired[str | None]
    access_token: NotRequired[str | None]
    planning_hints: NotRequired[str]

class OuterState(TypedDict):
    """State for the full competition lifecycle in the outer loop."""
    should_continue: Required[bool]

    session_id: NotRequired[str | None]
    access_token: NotRequired[str | None]
    current_task: NotRequired[dict[str, Any] | None]
    planning_hints: NotRequired[str]
    task_result: NotRequired[dict[str, Any] | None]
    error: NotRequired[str | None]