from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class TaskClassification(BaseModel):
    task_type: Literal["folder-organisation", "question-answering"] = Field(
        description="Must be 'folder-organisation' or 'question-answering'."
    )


class PlanningHintsResponse(BaseModel):
    hints: List[str] = Field(default_factory=list)
    caution: str = Field(default="")


class FileSummaryResponse(BaseModel):
    summary: str = Field(default="")


class QAAnswerSchema(BaseModel):
    answer: str = Field(default="")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="")


class QAActionSchema(BaseModel):
    """QA action schema supporting either tool-call or final answer path."""

    needs_image_analysis: bool = Field(
        default=False,
        description="True when the model requires image analysis before final answer.",
    )
    target_image_ids: List[str] = Field(
        default_factory=list,
        description="Image IDs selected from IMAGE_PLACEHOLDER tokens.",
    )
    vision_prompt: str = Field(
        default="",
        description="Prompt/question to ask the vision tool about selected images.",
    )
    answer: str = Field(default="")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="")


class ActionPlanResponse(BaseModel):
    answers: List[str] = Field(default_factory=list)
    thought_log: str = Field(default="")
    used_tools: List[str] = Field(default_factory=lambda: ["document_parser", "llm_client"])
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class FileSortDecision(BaseModel):
    file_path: str = Field(description="The exact file path or name to be sorted.")
    selected_folder: str = Field(
        description="The selected folder. MUST exactly match one item from the VALID_FOLDERS list."
    )
    reasoning: str = Field(description="Brief explanation of why this file belongs to the selected folder.")


class SortActionResponse(BaseModel):
    decisions: List[FileSortDecision] = Field(
        default_factory=list,
        description="List of sorting decisions for the provided files.",
    )
    overall_thought_log: str = Field(default="", description="Overall reasoning process for this task.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0.")


class VerificationResponse(BaseModel):
    answers: List[str] = Field(default_factory=list)
    thought_log: str = Field(default="")
    used_tools: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    changed: bool = Field(default=False)


class MetadataItem(BaseModel):
    key: str = Field(default="")
    value: str = Field(default="")


class ExtractedDocumentData(BaseModel):
    metadata: List[MetadataItem] = Field(default_factory=list)
    summary: str = Field(default="")