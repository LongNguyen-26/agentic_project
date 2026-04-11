# prompts/user_prompt.py
from typing import Any, Dict
from agent.prompts.sys_prompts import VALID_FOLDERS_STR


def build_qa_action_prompt(
    prompt_template: str,
    context: str,
    feedback: str = "",
    planning_hints: str = "",
) -> str:
    prompt = (
        f"Task Type: question-answering\n\n<task_instruction>\n{prompt_template}\n</task_instruction>\n\n"
        "<qa_output_policy>\n"
        "- Answer only from provided context; do not infer missing facts.\n"
        "- If evidence is ambiguous or missing, state uncertainty explicitly and reduce confidence.\n"
        "- In thought_log, include concrete evidence anchors when available: file path, page, image ID, or exact phrase.\n"
        "</qa_output_policy>\n\n"
    )
    if planning_hints:
        prompt += f"<planning_hints>\n{planning_hints}\n</planning_hints>\n\n"
    prompt += f"<context>\n{context}\n</context>\n\n"
    if feedback:
        prompt += (
            f"<feedback_from_previous_error>\n{feedback}\nDO NOT REPEAT THIS ERROR."
            "\n</feedback_from_previous_error>\n\n"
        )
    return prompt


def build_sort_action_prompt(
    prompt_template: str,
    file_summaries: Dict[str, str],
    feedback: str = "",
    planning_hints: str = "",
) -> str:
    summary_lines = [
        f"[Summary #{idx}] path={path}\n{summary.strip()}"
        for idx, (path, summary) in enumerate(file_summaries.items(), start=1)
    ]
    summaries_block = "\n\n".join(summary_lines) if summary_lines else "No file summaries provided."

    prompt = (
        "Task Type: folder-organisation\n\n"
        f"<task_instruction>\n{prompt_template}\n</task_instruction>\n\n"
        "<valid_folders>\n"
        f"{VALID_FOLDERS_STR}\n"
        "</valid_folders>\n\n"
        "<file_summaries>\n"
        f"{summaries_block}\n"
        "</file_summaries>\n\n"
        "You are provided with a list of file paths and their concise summaries."
        " Generate precise sorting decisions and select only valid folders.\n\n"
    )
    if planning_hints:
        prompt += f"<planning_hints>\n{planning_hints}\n</planning_hints>\n\n"
    if feedback:
        prompt += (
            f"<feedback_from_previous_error>\n{feedback}\nDO NOT REPEAT THIS ERROR."
            "\n</feedback_from_previous_error>\n\n"
        )
    return prompt


def build_qa_verification_prompt(prompt_template: str, draft_answer: Dict[str, Any], context: str) -> str:
    return (
        f"<task_instruction>\n{prompt_template}\n</task_instruction>\n\n"
        f"<source_context>\n{context}\n</source_context>\n\n"
        f"<draft_answer_json>\n{draft_answer}\n</draft_answer_json>\n\n"
        "Verify faithfulness and schema correctness.\n"
        "Reject unsupported claims, add uncertainty when evidence is weak, and ensure thought_log cites concrete anchors where possible."
    )


def build_sort_verification_prompt(
    prompt_template: str,
    draft_answer: Dict[str, Any],
    file_summaries: Dict[str, str],
) -> str:
    summary_lines = [
        f"[Summary #{idx}] path={path}\n{summary.strip()}"
        for idx, (path, summary) in enumerate(file_summaries.items(), start=1)
    ]
    summaries_block = "\n\n".join(summary_lines) if summary_lines else "No file summaries provided."
    return (
        f"<task_instruction>\n{prompt_template}\n</task_instruction>\n\n"
        "<valid_folders>\n"
        f"{VALID_FOLDERS_STR}\n"
        "</valid_folders>\n\n"
        f"<file_summaries>\n{summaries_block}\n</file_summaries>\n\n"
        f"<draft_answer_json>\n{draft_answer}\n</draft_answer_json>\n\n"
        "Re-validate sorting output. Keep answers as [] for folder tasks and correct folder choices if unsupported."
    )


def build_task_classification_prompt(prompt_template: str) -> str:
    return f"Prompt Template to classify:\n{prompt_template}"


def build_planning_hints_prompt(prompt_template: str) -> str:
    return (
        "Read the task prompt and extract warnings for execution.\n"
        "Focus on: required output format, redacted placeholders, missing-value convention,"
        " possible calculations/units, and ambiguous terms.\n"
        "Return concise hints only, do not solve the task.\n\n"
        f"Task prompt:\n{prompt_template}"
    )


def build_hints_extraction_prompt(prompt_template: str) -> str:
    """Backward-compatible alias for planning hints prompt."""
    return build_planning_hints_prompt(prompt_template)


def build_file_summary_prompt(file_name: str, file_content: str) -> str:
    return (
        "Summarize the following document for context cache usage.\n"
        "Requirements:\n"
        "1. Produce 3-5 concise sentences capturing key content.\n"
        "2. List placeholder tokens like [tag_name] verbatim if present.\n"
        "3. Do not infer beyond document content.\n\n"
        f"[FILE NAME]\n{file_name}\n\n"
        f"[FILE CONTENT]\n{file_content}"
    )