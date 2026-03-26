# agent/nodes/inner_loop.py
from agent.prompts.sys_prompts import SYS_ACTION_GEN, SYS_VERIFY
from agent.prompts.user_prompt import build_action_prompt, build_verification_prompt
from agent.state import InnerState
from clients.competition_client import APIClient
from clients.llm_client import LLMService
from config import config
from core.logger import get_logger
from models.llm_schemas import QAAnswerSchema, VerificationResponse
from tools.context_manager import format_full_context
from tools.document_parser import parse_resource_bytes
from tools.rag_engine import build_and_retrieve_context

llm_service = None
api_client = None
logger = get_logger(__name__)


def _get_llm_service() -> LLMService:
    global llm_service
    if llm_service is None:
        llm_service = LLMService()
    return llm_service


def _get_api_client() -> APIClient:
    global api_client
    if api_client is None:
        api_client = APIClient()
    return api_client

def observability_node(state: InnerState) -> dict:
    """Bước 1: Tải tài nguyên task và parse thành text."""
    logger.info(
        "[Observability] task_id=%s type=%s resources=%s",
        state.get("task_id"),
        state.get("task_type"),
        len(state.get("resources", [])),
    )

    parsed_documents = []
    full_text_parts = []
    resources = state.get("resources", [])
    local_client = _get_api_client()
    if state.get("session_id"):
        local_client.session_id = state["session_id"]
    if state.get("access_token"):
        local_client.access_token = state["access_token"]

    for resource in resources:
        file_path = resource.get("file_path", "unknown")
        token = resource.get("token")
        if not token:
            continue
        downloaded = local_client.download_and_persist_resource(
            task_id=state["task_id"],
            file_path=file_path,
            token=token,
        )
        text = parse_resource_bytes(file_path, downloaded["bytes"])
        parsed_documents.append({"file_path": file_path, "text": text})
        full_text_parts.append(f"[FILE] {file_path}\n{text}")

    full_text = "\n\n".join(full_text_parts).strip()
    should_use_rag = state["task_type"] == "question-answering" and len(resources) >= config.QA_RAG_DOC_THRESHOLD
    return {
        "parsed_documents": parsed_documents,
        "parsed_text": full_text,
        "use_rag": should_use_rag,
        "used_tools": ["document_parser"],
    }

def setup_rag_node(state: InnerState) -> dict:
    """Khi QA có nhiều tài liệu, dùng RAG để lấy ngữ cảnh trọng tâm."""
    logger.info("[Context] Using RAG retrieval for task_id=%s", state.get("task_id"))
    query = state.get("prompt_template", "")
    context = build_and_retrieve_context(state["parsed_text"], query, top_k=config.RAG_TOP_K)
    used_tools = list(state.get("used_tools", []))
    used_tools.append("rag_engine")
    return {"retrieved_context": context, "used_tools": used_tools}

def setup_context_manager_node(state: InnerState) -> dict:
    """Nếu không cần RAG thì dùng toàn văn bản đã chuẩn hóa."""
    logger.info("[Context] Using full context manager for task_id=%s", state.get("task_id"))
    context = format_full_context(state["parsed_text"])
    used_tools = list(state.get("used_tools", []))
    used_tools.append("context_manager")
    return {"retrieved_context": context, "used_tools": used_tools}

def action_generation_node(state: InnerState) -> dict:
    """Bước 2: Sinh đáp án structured output thay cho regex parsing."""
    logger.info(
        "[Action] Generating response task_id=%s attempt=%s",
        state.get("task_id"),
        state.get("attempts", 0) + 1,
    )

    context = state["retrieved_context"]
    feedback = state.get("verification_feedback", "")

    if state["task_type"] == "folder-organisation":
        file_contents = {doc["file_path"]: doc["text"] for doc in state.get("parsed_documents", [])}
        response = _get_llm_service().generate_folder_response(
            prompt_template=state.get("prompt_template", ""),
            file_contents=file_contents,
        )
        return {
            "draft_answer": response,
            "action_plan": response,
            "confidence_score": float(response.get("confidence", 0.0)),
        }

    prompt = build_action_prompt(
        task_type=state["task_type"],
        prompt_template=state.get("prompt_template", ""),
        context=context,
        feedback=feedback,
    )

    draft_response = _get_llm_service().generate_structured(
        system_prompt=SYS_ACTION_GEN,
        user_prompt=prompt,
        response_model=QAAnswerSchema,
    )
    response_payload = {
        "answers": [draft_response.answer.strip()],
        "thought_log": draft_response.reasoning.strip(),
        "used_tools": state.get("used_tools", []),
        "confidence": float(draft_response.confidence),
    }
    return {
        "draft_answer": response_payload,
        "action_plan": response_payload,
        "confidence_score": float(draft_response.confidence),
    }

def verifiability_node(state: InnerState) -> dict:
    """Bước 3: Tự kiểm duyệt và tự sửa nếu cần."""
    logger.info("[Verifiability] Checking answer consistency for task_id=%s", state.get("task_id"))

    if state["task_type"] == "folder-organisation":
        return {
            "is_verified": True,
            "verification_feedback": "",
            "attempts": state["attempts"] + 1,
        }

    draft = state["draft_answer"]
    context = state["retrieved_context"]
    prompt = build_verification_prompt(
        prompt_template=state.get("prompt_template", ""),
        draft_answer=draft,
        context=context,
    )
    verification = _get_llm_service().generate_structured(
        system_prompt=SYS_VERIFY,
        user_prompt=prompt,
        response_model=VerificationResponse,
        max_completion_tokens=config.VERIFICATION_MAX_OUTPUT_TOKENS,
    )

    changed = verification.changed
    confidence = float(verification.confidence)
    next_answer = draft
    if changed or confidence < config.VERIFIER_MIN_CONFIDENCE:
        next_answer = {
            "answers": verification.answers or draft.get("answers", []),
            "thought_log": verification.thought_log or draft.get("thought_log", ""),
            "used_tools": verification.used_tools or draft.get("used_tools", []),
            "confidence": confidence,
        }

    return {
        "draft_answer": next_answer,
        "confidence_score": confidence,
        "is_verified": confidence >= config.VERIFIER_MIN_CONFIDENCE,
        "verification_feedback": verification.thought_log,
        "attempts": state["attempts"] + 1,
    }