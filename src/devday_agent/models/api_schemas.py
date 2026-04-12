from typing import Any, Dict, List

from pydantic import BaseModel


class SessionRequest(BaseModel):
    session_id: str | None = None


class SessionResponse(BaseModel):
    session_id: str
    agent_id: str
    access_token: str
    token_type: str
    expires_in: int


class Resource(BaseModel):
    file_path: str
    token: str


class TaskResponse(BaseModel):
    task_id: str
    phase: int | None = None
    type: str | None = None
    prompt_template: str
    resources: List[Resource]


class SubmissionResponse(BaseModel):
    task_id: str
    session_id: str
    total_files: int
    correct: int
    score: float
    details: List[Dict[str, Any]]