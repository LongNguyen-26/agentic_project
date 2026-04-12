# agent/nodes/inner_loop.py
import re
from typing import Any, Dict, List, Set
from collections import Counter

from devday_agent.agent.prompts.sys_prompts import (
    SYS_ACTION_QA,
    SYS_ACTION_SORT,
    SYS_VERIFY_QA,
    SYS_VERIFY_SORT,
    VALID_FOLDERS_STR,
)
from devday_agent.agent.prompts.user_prompt import (
    build_qa_action_prompt,
    build_qa_verification_prompt,
    build_sort_action_prompt,
    build_sort_verification_prompt,
)
from devday_agent.agent.state import InnerState
from devday_agent.clients.competition_client import APIClient
from devday_agent.clients.llm_client import LLMService
from config import config
from devday_agent.core.logger import get_logger
from devday_agent.core.checkpoint import load_parsed_text_cache, save_parsed_text_cache
from devday_agent.models.llm_schemas import QAActionSchema, SortActionResponse, VerificationResponse
from tools.context_manager import format_context_from_documents, format_full_context, get_or_create_file_summary
from tools.document_parser import parse_resource_bytes
from tools.rag_engine import build_and_retrieve_context
from tools.vision_tool import analyze_images_from_cache

llm_service = None
api_client = None
logger = get_logger(__name__)


def _parse_valid_folders(raw_text: str) -> Set[str]:
    folders: Set[str] = set()
    for line in raw_text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if "." in cleaned:
            _, maybe_name = cleaned.split(".", 1)
            cleaned = maybe_name.strip()
        if cleaned:
            folders.add(cleaned)
    return folders


VALID_FOLDERS: Set[str] = _parse_valid_folders(VALID_FOLDERS_STR)
FOLDER_LINE_PATTERN = re.compile(r"^\s*Folder:\s*(.+?)\s*$", re.IGNORECASE)
SORT_FILE_LINE_PATTERN = re.compile(r"^\s*-\s*File:\s*(.+?)\s*$", re.IGNORECASE)
IMAGE_ID_PATTERN = re.compile(
    r"\[IMAGE_PLACEHOLDER\s*\|\s*ID:\s*([A-Za-z0-9_-]+)\s*\|",
    re.IGNORECASE,
)
FILE_PATH_MARKER_PATTERN = re.compile(r"Public/[^\s\]\)\'\"]+", re.IGNORECASE)
PAGE_MARKER_PATTERN = re.compile(r"(?:page\s*\d+|ページ\s*\d+)", re.IGNORECASE)
IMAGE_ID_LITERAL_PATTERN = re.compile(r"\b[0-9a-f]{16}\b", re.IGNORECASE)

FORCED_VISION_KEYWORDS = (
    "anh",
    "ảnh",
    "hinh",
    "hình",
    "image",
    "images",
    "photo",
    "photos",
    "picture",
    "visual",
    "図面",
    "写真",
    "画像",
    "配線図",
)
FORCED_VISION_MAX_IMAGES = 8


def _extract_sort_entries_from_thought_log(thought_log: str) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    current_file = ""

    for line in thought_log.splitlines():
        file_match = SORT_FILE_LINE_PATTERN.match(line)
        if file_match:
            current_file = file_match.group(1).strip()
            continue

        folder_match = FOLDER_LINE_PATTERN.match(line)
        if folder_match and current_file:
            entries.append(
                {
                    "file_path": current_file,
                    "selected_folder": folder_match.group(1).strip(),
                }
            )
            current_file = ""

    return entries


def _should_force_vision(prompt_template: str) -> bool:
    prompt_lower = (prompt_template or "").lower()
    return any(keyword in prompt_lower for keyword in FORCED_VISION_KEYWORDS)


def _extract_image_ids_from_text(raw_text: str) -> List[str]:
    if not raw_text:
        return []
    return [match.group(1).strip() for match in IMAGE_ID_PATTERN.finditer(raw_text)]


def _collect_candidate_image_ids(state: InnerState) -> List[str]:
    ordered_ids: List[str] = []
    seen: Set[str] = set()

    # Prefer image placeholders that survived retrieval, then fall back to full parsed docs.
    text_sources: List[str] = [state.get("retrieved_context", "")]
    for document in state.get("parsed_documents", []):
        text_sources.append(document.get("text", ""))

    for raw_text in text_sources:
        for image_id in _extract_image_ids_from_text(raw_text):
            if image_id and image_id not in seen:
                seen.add(image_id)
                ordered_ids.append(image_id)

    return ordered_ids


def _collect_output_markers(output_text: str) -> Set[str]:
    markers: Set[str] = set()
    if not output_text:
        return markers

    markers.update(FILE_PATH_MARKER_PATTERN.findall(output_text))
    markers.update(PAGE_MARKER_PATTERN.findall(output_text))
    markers.update(IMAGE_ID_LITERAL_PATTERN.findall(output_text))
    markers.update(_extract_image_ids_from_text(output_text))
    return {marker.strip() for marker in markers if marker.strip()}


def _normalize_marker(marker: str) -> str:
    normalized = " ".join(str(marker).strip().lower().split())
    return normalized.replace("\\", "/")


def _collect_grounding_anchor_pool(context_text: str, parsed_documents: List[Dict[str, Any]]) -> Set[str]:
    anchors: Set[str] = set()

    def _add_anchor(value: str) -> None:
        normalized = _normalize_marker(value)
        if normalized:
            anchors.add(normalized)

    for marker in _collect_output_markers(context_text or ""):
        _add_anchor(marker)

    for document in parsed_documents or []:
        file_path = str(document.get("file_path", "")).strip()
        if file_path:
            _add_anchor(file_path)

        raw_text = str(document.get("text", ""))
        if not raw_text:
            continue

        for marker in FILE_PATH_MARKER_PATTERN.findall(raw_text):
            _add_anchor(marker)
        for marker in PAGE_MARKER_PATTERN.findall(raw_text):
            _add_anchor(marker)
        for image_id in _extract_image_ids_from_text(raw_text):
            _add_anchor(image_id)

    return anchors


def _count_grounded_markers(
    output_text: str,
    context_text: str,
    parsed_documents: List[Dict[str, Any]],
) -> tuple[int, int]:
    raw_markers = _collect_output_markers(output_text)
    if not raw_markers:
        return 0, 0

    markers = {_normalize_marker(marker) for marker in raw_markers if _normalize_marker(marker)}
    if not markers:
        return 0, 0

    anchor_pool = _collect_grounding_anchor_pool(context_text, parsed_documents)
    normalized_context = _normalize_marker(context_text or "")

    grounded = 0
    for marker in markers:
        if marker in anchor_pool or marker in normalized_context:
            grounded += 1

    return grounded, len(markers)


def _append_answer_log(state: InnerState, answers: List[str], confidence: float, grounded: int, detected: int) -> str:
    attempt = int(state.get("attempts", 0)) + 1
    preview_answers = [ans.strip()[:600] for ans in answers if ans and ans.strip()]
    lines = [f"[attempt {attempt}] confidence={confidence:.3f} grounding={grounded}/{detected}"]
    for ans in preview_answers:
        lines.append(f"- {ans}")

    entry = "\n".join(lines)
    previous = (state.get("answer_log") or "").strip()
    merged = f"{previous}\n\n{entry}" if previous else entry
    max_chars = max(int(config.QA_ANSWER_LOG_MAX_CHARS), 1000)
    return merged[-max_chars:]


def _build_grounding_fallback_answer(
    draft_answer: Dict[str, Any],
    grounded_count: int,
    detected_count: int,
    required_count: int,
) -> Dict[str, Any]:
    tentative_answers = draft_answer.get("answers", [])
    tentative = ""
    if tentative_answers:
        tentative = str(tentative_answers[0]).strip()

    if tentative:
        conservative_answer = tentative
    else:
        conservative_answer = "Insufficient evidence in provided context."

    prior_thought = str(draft_answer.get("thought_log", "")).strip()
    fallback_note = (
        "[Fallback] Exhausted retries due to grounding failure. "
        f"[Grounding Diagnostics] grounded={grounded_count}, detected={detected_count}, required={required_count}."
    )
    fallback_thought = f"{prior_thought}\n\n{fallback_note}" if prior_thought else fallback_note

    return {
        "answers": [conservative_answer],
        "thought_log": fallback_thought,
        "used_tools": draft_answer.get("used_tools", []),
        "confidence": float(draft_answer.get("confidence", 0.0) or 0.0),
    }


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


def _generate_sort_action(state: InnerState, feedback: str) -> dict:
    file_summaries = {doc["file_path"]: doc["summary"] for doc in state.get("parsed_documents", [])}
    prompt = build_sort_action_prompt(
        prompt_template=state.get("prompt_template", ""),
        file_summaries=file_summaries,
        feedback=feedback,
        planning_hints=state.get("planning_hints", ""),
    )
    draft_response = _get_llm_service().generate_action_response(
        system_prompt=SYS_ACTION_SORT,
        user_prompt=prompt,
        response_model=SortActionResponse,
        max_completion_tokens=config.SORT_ACTION_MAX_OUTPUT_TOKENS,
        retry_max_output_tokens=max(
            config.SORT_ACTION_RETRY_MAX_OUTPUT_TOKENS,
            config.SORT_ACTION_MAX_OUTPUT_TOKENS,
        ),
    )

    formatted_thought_log = (draft_response.overall_thought_log or "").strip()
    if formatted_thought_log:
        formatted_thought_log += "\n\nSorting Details:\n"
    else:
        formatted_thought_log = "Sorting Details:\n"

    for decision in draft_response.decisions:
        formatted_thought_log += (
            f"- File: {decision.file_path}\n"
            f"  Folder: {decision.selected_folder}\n"
            f"  Reasoning: {decision.reasoning}\n\n"
        )

    response_payload = {
        "answers": [],
        "thought_log": formatted_thought_log.strip(),
        "used_tools": state.get("used_tools", []),
        "confidence": float(draft_response.confidence),
    }
    return {
        "draft_answer": response_payload,
        "action_plan": response_payload,
        "confidence_score": float(draft_response.confidence),
    }


def _generate_qa_action(state: InnerState, feedback: str) -> dict:
    context = state["retrieved_context"]
    if state.get("tool_observations"):
        prior_observations = "\n".join(state.get("tool_observations", []))
        context = (
            f"{context}\n\n"
            "[Vision Tool Observations]\n"
            f"{prior_observations}"
        )

    # Force at least one vision pass for image-heavy prompts before finalizing QA answer.
    if (
        _should_force_vision(state.get("prompt_template", ""))
        and not state.get("tool_observations")
    ):
        candidate_image_ids = _collect_candidate_image_ids(state)
        target_ids = candidate_image_ids[:FORCED_VISION_MAX_IMAGES]
        if target_ids:
            logger.info(
                "[Action] Forcing vision route task_id=%s image_count=%s",
                state.get("task_id"),
                len(target_ids),
            )
            forced_prompt = (
                "Analyze the selected image(s) to answer this task. "
                "Extract only visual evidence relevant to the prompt and quote visible labels/text when possible.\n\n"
                f"Task:\n{state.get('prompt_template', '')}"
            )
            return {
                "tool_calls": target_ids,
                "vision_prompt": forced_prompt,
                "confidence_score": max(float(state.get("confidence_score", 0.0) or 0.0), 0.5),
            }
        logger.info(
            "[Action] Prompt matched forced-vision keywords but no IMAGE_PLACEHOLDER IDs found task_id=%s",
            state.get("task_id"),
        )

    prompt = build_qa_action_prompt(
        prompt_template=state.get("prompt_template", ""),
        context=context,
        feedback=feedback,
        planning_hints=state.get("planning_hints", ""),
    )

    draft_response = _get_llm_service().generate_action_response(
        system_prompt=SYS_ACTION_QA,
        user_prompt=prompt,
        response_model=QAActionSchema,
        max_completion_tokens=config.QA_ACTION_MAX_OUTPUT_TOKENS,
        retry_max_output_tokens=max(
            config.QA_ACTION_RETRY_MAX_OUTPUT_TOKENS,
            config.QA_ACTION_MAX_OUTPUT_TOKENS,
        ),
    )

    if draft_response.requires_vision_tool:
        target_ids = [img_id.strip() for img_id in draft_response.target_image_ids if img_id.strip()]
        return {
            "tool_calls": target_ids,
            "vision_prompt": draft_response.vision_prompt.strip(),
            "confidence_score": float(draft_response.confidence),
        }

    if draft_response.needs_image_analysis:
        logger.warning("[Action] needs_image_analysis=True but no target_image_ids were returned")

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
        "tool_calls": [],
        "vision_prompt": "",
    }


def _verify_sort(state: InnerState) -> dict:
    draft = state["draft_answer"]
    file_summaries = {doc["file_path"]: doc.get("summary", "") for doc in state.get("parsed_documents", [])}
    expected_files = [
        str(doc.get("file_path", "")).strip()
        for doc in state.get("parsed_documents", [])
        if str(doc.get("file_path", "")).strip()
    ]
    expected_file_set = set(expected_files)
    prompt = build_sort_verification_prompt(
        prompt_template=state.get("prompt_template", ""),
        draft_answer=draft,
        file_summaries=file_summaries,
    )
    verification = _get_llm_service().generate_verification_response(
        system_prompt=SYS_VERIFY_SORT,
        user_prompt=prompt,
        response_model=VerificationResponse,
        max_completion_tokens=config.SORT_VERIFICATION_MAX_OUTPUT_TOKENS,
        retry_max_output_tokens=max(
            config.SORT_VERIFICATION_RETRY_MAX_OUTPUT_TOKENS,
            config.SORT_VERIFICATION_MAX_OUTPUT_TOKENS,
        ),
    )

    confidence = float(verification.confidence)

    original_thought_log = draft.get("thought_log", "").strip()
    verifier_log = verification.thought_log.strip()

    if verifier_log:
        verified_thought_log = f"{original_thought_log}\n\n[Verification]: {verifier_log}"
    else:
        verified_thought_log = original_thought_log

    next_answer = {
        "answers": [],
        "thought_log": verified_thought_log,
        "used_tools": verification.used_tools or draft.get("used_tools", []),
        "confidence": confidence,
    }

    # Validate file-folder decisions from draft sorting details only; verifier notes are free text.
    sort_entries = _extract_sort_entries_from_thought_log(draft.get("thought_log", ""))
    selected_files = [entry["file_path"] for entry in sort_entries]
    selected_folders = [entry["selected_folder"] for entry in sort_entries]
    invalid_folders = sorted({folder for folder in selected_folders if folder not in VALID_FOLDERS})
    selected_file_set = set(selected_files)

    file_counter = Counter(selected_files)
    duplicated_files = sorted({file_path for file_path, count in file_counter.items() if count > 1})
    missing_files = sorted(expected_file_set - selected_file_set)
    unexpected_files = sorted(selected_file_set - expected_file_set)

    if invalid_folders:
        feedback = (
            "Invalid folder(s) detected: "
            f"{', '.join(invalid_folders)}. "
            "Use only folders listed in VALID_FOLDERS_STR."
        )
        is_valid = False
    elif duplicated_files:
        feedback = (
            "Duplicate file decision(s) detected: "
            f"{', '.join(duplicated_files)}. "
            "Each resource must be assigned exactly one folder."
        )
        is_valid = False
    elif missing_files:
        feedback = (
            "Missing folder decision(s) for file(s): "
            f"{', '.join(missing_files)}. "
            "Provide one folder decision for every resource."
        )
        is_valid = False
    elif unexpected_files:
        feedback = (
            "Unexpected file decision(s) detected: "
            f"{', '.join(unexpected_files)}. "
            "Only files present in task resources are allowed."
        )
        is_valid = False
    else:
        is_valid = (
            len(sort_entries) == len(expected_files)
            and bool(selected_folders)
            and confidence >= config.VERIFIER_MIN_CONFIDENCE
        )
        feedback = ""
        if not is_valid:
            feedback = (
                f"Folder verification failed: decisions={len(sort_entries)}/{len(expected_files)}, "
                f"confidence={confidence:.3f}, required_confidence={config.VERIFIER_MIN_CONFIDENCE:.3f}."
            )

    return {
        "draft_answer": next_answer,
        "confidence_score": confidence,
        "is_verified": is_valid,
        "verification_feedback": feedback,
        "attempts": state["attempts"] + 1,
    }


def _verify_qa(state: InnerState) -> dict:
    draft = state["draft_answer"]
    context = state["retrieved_context"]
    if state.get("tool_observations"):
        prior_observations = "\n".join(state.get("tool_observations", []))
        context = (
            f"{context}\n\n"
            "[Vision Tool Observations]\n"
            f"{prior_observations}"
        )

    prompt = build_qa_verification_prompt(
        prompt_template=state.get("prompt_template", ""),
        draft_answer=draft,
        context=context,
    )
    verification = _get_llm_service().generate_verification_response(
        system_prompt=SYS_VERIFY_QA,
        user_prompt=prompt,
        response_model=VerificationResponse,
        max_completion_tokens=config.QA_VERIFICATION_MAX_OUTPUT_TOKENS,
        retry_max_output_tokens=max(
            config.QA_VERIFICATION_RETRY_MAX_OUTPUT_TOKENS,
            config.QA_VERIFICATION_MAX_OUTPUT_TOKENS,
        ),
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

    output_text = "\n".join(next_answer.get("answers", [])) + "\n" + next_answer.get("thought_log", "")
    grounded_count, detected_count = _count_grounded_markers(
        output_text,
        context,
        state.get("parsed_documents", []),
    )
    grounding_pass = True
    grounding_feedback = ""
    required = max(int(config.QA_GROUNDING_MIN_EVIDENCE_MARKERS), 0)
    if bool(config.QA_GROUNDING_ENFORCED):
        if detected_count == 0 or grounded_count < required:
            grounding_pass = False
            grounding_feedback = (
                f"Grounding evidence is insufficient: grounded={grounded_count}, detected={detected_count}, "
                f"required={required}. Cite explicit file path/page/image id anchors from context."
            )

    answer_log = _append_answer_log(
        state,
        next_answer.get("answers", []),
        confidence,
        grounded_count,
        detected_count,
    )

    combined_feedback = (verification.thought_log or "").strip()
    if grounding_feedback:
        combined_feedback = f"{combined_feedback}\n\n[Grounding]: {grounding_feedback}".strip()

    next_attempt = state["attempts"] + 1
    fallback_due_to_grounding = False
    if bool(config.QA_GROUNDING_ENFORCED) and not grounding_pass and next_attempt >= max(int(config.MAX_RETRIES), 1):
        next_answer = _build_grounding_fallback_answer(next_answer, grounded_count, detected_count, required)
        fallback_due_to_grounding = True
        combined_feedback = (
            f"{combined_feedback}\n\n"
            "[Fallback]: Max retries reached with grounding failure; switched to conservative answer while keeping diagnostics in thought_log."
        ).strip()

    return {
        "draft_answer": next_answer,
        "confidence_score": confidence,
        "is_verified": confidence >= config.VERIFIER_MIN_CONFIDENCE and grounding_pass,
        "verification_feedback": combined_feedback,
        "answer_log": answer_log,
        "attempts": next_attempt,
        "fallback_due_to_grounding": fallback_due_to_grounding,
    }

def observability_node(state: InnerState) -> dict:
    """Collect resources, parse content, and prepare source context.

    Args:
        state: Inner-loop state containing task metadata, credentials, and resources.

    Returns:
        dict: Updated keys including parsed_documents, parsed_text, use_rag, and used_tools.
    """
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

    if not hasattr(observability_node, "_in_memory_cache"):
        observability_node._in_memory_cache = {}

    for resource in resources:
        file_path = resource.get("file_path", "unknown")
        token = resource.get("token")
        if not token:
            continue

        if file_path in observability_node._in_memory_cache:
            logger.info("[Observability] RAM Cache HIT: %s", file_path)
            text = observability_node._in_memory_cache[file_path]
        else:
            cached_text = load_parsed_text_cache(file_path)
            if cached_text is not None:
                logger.info("[Observability] Disk Cache HIT: %s", file_path)
                text = cached_text
                observability_node._in_memory_cache[file_path] = text
            else:
                logger.info("[Observability] Cache MISS, downloading & parsing: %s", file_path)
                downloaded = local_client.download_and_persist_resource(
                    task_id=state["task_id"],
                    file_path=file_path,
                    token=token,
                )
                text = parse_resource_bytes(file_path, downloaded["bytes"])
                observability_node._in_memory_cache[file_path] = text
                save_parsed_text_cache(file_path, text)

        # Continue file-summary flow using cache-aware summarization.
        summary, cache_hit = get_or_create_file_summary(
            file_path=file_path,
            raw_text=text,
            llm_service=_get_llm_service(),
        )
        parsed_documents.append(
            {
                "file_path": file_path,
                "text": text,
                "summary": summary,
                "summary_cache_hit": cache_hit,
            }
        )
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
    """Build condensed retrieval context for multi-document QA tasks.

    Args:
        state: Inner-loop state with parsed_documents and prompt_template.

    Returns:
        dict: Retrieved context and updated used_tools list.
    """
    logger.info("[Context] Using RAG retrieval for task_id=%s", state.get("task_id"))
    query = state.get("prompt_template", "")
    
    # Pass parsed_documents instead of raw parsed_text.
    parsed_documents = state.get("parsed_documents", [])
    context = build_and_retrieve_context(parsed_documents, query, top_k=config.RAG_TOP_K)
    
    used_tools = list(state.get("used_tools", []))
    used_tools.append("rag_engine")
    return {"retrieved_context": context, "used_tools": used_tools}

def setup_context_manager_node(state: InnerState) -> dict:
    """Build context from full parsed data when RAG is not used.

    Args:
        state: Inner-loop state containing parsed_documents or parsed_text.

    Returns:
        dict: Full context and updated used_tools list.
    """
    logger.info("[Context] Using full context manager for task_id=%s", state.get("task_id"))
    context = format_context_from_documents(state.get("parsed_documents", []))
    if not context.strip():
        context = format_full_context(state["parsed_text"])
    used_tools = list(state.get("used_tools", []))
    used_tools.append("context_manager")
    return {"retrieved_context": context, "used_tools": used_tools}

def action_generation_node(state: InnerState) -> dict:
    """Generate draft output by task_type for the verification stage.

    Args:
        state: Inner-loop state with context, prompt, and verifier feedback.

    Returns:
        dict: draft_answer, action_plan, and confidence_score for the next step.
    """
    logger.info(
        "[Action] Generating response task_id=%s attempt=%s",
        state.get("task_id"),
        state.get("attempts", 0) + 1,
    )

    feedback = state.get("verification_feedback", "")

    if state["task_type"] == "folder-organisation":
        return _generate_sort_action(state, feedback)

    return _generate_qa_action(state, feedback)


def vision_tool_node(state: InnerState) -> dict:
    """Execute vision analysis for requested image IDs and append observation.

    Args:
        state: Inner-loop state containing tool_calls and vision_prompt.

    Returns:
        dict: Updates tool_observations and clears tool_calls to avoid infinite loop.
    """
    image_ids = [img_id.strip() for img_id in state.get("tool_calls", []) if img_id.strip()]
    vision_prompt = state.get("vision_prompt", "")

    if not image_ids:
        return {
            "tool_calls": [],
            "vision_prompt": "",
        }

    logger.info(
        "[VisionToolNode] task_id=%s image_count=%s",
        state.get("task_id"),
        len(image_ids),
    )

    try:
        observation = analyze_images_from_cache(image_ids=image_ids, vision_prompt=vision_prompt)
    except Exception as exc:
        observation = f"[Vision Tool] ERROR: unexpected failure while analyzing images: {exc}"
        logger.warning(observation)

    existing_observations = list(state.get("tool_observations", []))
    existing_observations.append(observation)

    used_tools = list(state.get("used_tools", []))
    if "vision_tool" not in used_tools:
        used_tools.append("vision_tool")

    return {
        "tool_observations": existing_observations,
        "tool_calls": [],
        "vision_prompt": "",
        "used_tools": used_tools,
    }

def verifiability_node(state: InnerState) -> dict:
    """Self-verify the draft answer and decide pass/retry status.

    Args:
        state: Inner-loop state with draft_answer, context, and retry counters.

    Returns:
        dict: Updated draft_answer, confidence_score, is_verified, feedback, and attempts.
    """
    logger.info("[Verifiability] Checking answer consistency for task_id=%s", state.get("task_id"))

    if state["task_type"] == "folder-organisation":
        return _verify_sort(state)

    return _verify_qa(state)