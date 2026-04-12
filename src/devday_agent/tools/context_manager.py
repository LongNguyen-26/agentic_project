# tools/context_manager.py

import hashlib
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from config import config
from devday_agent.core.checkpoint import load_file_summary_cache, save_file_summary_cache
from devday_agent.core.logger import get_logger
from devday_agent.models.llm_schemas import FileSummaryResponse


logger = get_logger(__name__)

# Cache key = "{file_path}::{sha1(raw_text)}"
_file_summary_cache: Dict[str, Dict[str, Any]] = {}
_cache_loaded = False


def _ensure_cache_loaded() -> None:
    """Load persisted cross-task cache once per process."""
    global _cache_loaded
    if _cache_loaded:
        return
    loaded = load_file_summary_cache()
    if isinstance(loaded, dict):
        for key, value in loaded.items():
            if isinstance(value, dict) and "summary" in value:
                _file_summary_cache[key] = value
    _cache_loaded = True
    logger.info("[context] Loaded file summary cache entries=%d", len(_file_summary_cache))


def _persist_cache() -> None:
    """Persist in-memory summary cache into session checkpoint."""
    save_file_summary_cache(_file_summary_cache)


def _fingerprint(text: str) -> str:
    """Build stable fingerprint for cache keying."""
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _cache_key(file_path: str, raw_text: str) -> str:
    """Create deterministic cache key for a file content snapshot."""
    return f"{file_path}::{_fingerprint(raw_text)}"


def _clean_text(raw_text: str) -> str:
    """Normalize whitespace to reduce prompt token usage but PRESERVE NEWLINES."""
    # Collapse repeated spaces/tabs without removing newline boundaries.
    return re.sub(r'[ \t]+', ' ', raw_text)


def _fallback_summary(file_path: str, raw_text: str) -> str:
    """Deterministic fallback summary when LLM summary generation fails."""
    cleaned = _clean_text(raw_text)
    preview = cleaned[:500]
    char_count = len(cleaned)
    return (
        f"Document {file_path} was parsed successfully. "
        f"Normalized text length: {char_count} characters. "
        f"Content preview: {preview}"
    )


def get_cached_file_summary(file_path: str, raw_text: str) -> Optional[str]:
    """Return cached summary when available for the exact file fingerprint."""
    _ensure_cache_loaded()
    key = _cache_key(file_path, raw_text)
    item = _file_summary_cache.get(key)
    if not item:
        return None
    return str(item.get("summary", "")).strip() or None


def get_or_create_file_summary(
    *,
    file_path: str,
    raw_text: str,
    llm_service: Any,
) -> Tuple[str, bool]:
    """Get cached file summary or create a fresh 3-5 sentence summary.

    Returns:
        Tuple(summary, cache_hit)
    """
    cached = get_cached_file_summary(file_path=file_path, raw_text=raw_text)
    if cached:
        return cached, True

    cleaned = _clean_text(raw_text)
    if not cleaned:
        summary = f"Document {file_path} does not contain extractable text to summarize."
    elif llm_service is None:
        summary = _fallback_summary(file_path=file_path, raw_text=cleaned)
    else:
        system_prompt = (
            "You summarize construction documents for token-efficient downstream QA. "
            "Output one compact summary paragraph with 3-5 sentences."
        )
        user_prompt = (
            f"File path: {file_path}\n"
            "Generate a concise summary containing document metadata clues, key entities, "
            "numbers/dates, and the main purpose. Keep factual and avoid speculation.\n\n"
            f"Content:\n{cleaned[:config.FILE_SUMMARY_SOURCE_MAX_CHARS]}"
        )
        try:
            response = llm_service.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=FileSummaryResponse,
                max_completion_tokens=max(config.FILE_SUMMARY_MAX_OUTPUT_TOKENS, 256),
                retry_max_output_tokens=max(
                    config.FILE_SUMMARY_RETRY_MAX_OUTPUT_TOKENS,
                    config.FILE_SUMMARY_MAX_OUTPUT_TOKENS,
                ),
            )
            summary = response.summary.strip() or _fallback_summary(file_path=file_path, raw_text=cleaned)
        except Exception:
            logger.warning("[context] Summary generation failed for %s; using fallback", file_path, exc_info=True)
            summary = _fallback_summary(file_path=file_path, raw_text=cleaned)

    key = _cache_key(file_path, raw_text)
    _file_summary_cache[key] = {
        "file_path": file_path,
        "summary": summary,
        "updated_at": int(time.time()),
    }
    _persist_cache()
    return summary, False


def format_full_context(raw_text: str) -> str:
    """Format full raw text context for non-RAG processing."""
    cleaned_text = _clean_text(raw_text)
    return (
        "[BEGIN RAW SOURCE DOCUMENT]\n"
        f"{cleaned_text}\n"
        "[END RAW SOURCE DOCUMENT]"
    )


def format_context_from_documents(parsed_documents: List[Dict[str, Any]]) -> str:
    """Build compact prompt context from per-file summaries when available."""
    blocks: List[str] = []
    for doc in parsed_documents:
        file_path = doc.get("file_path", "unknown")
        summary = (doc.get("summary") or "").strip()
        if summary:
            blocks.append(f"[FILE] {file_path}\n[SUMMARY]\n{summary}")
            continue
        text = _clean_text(str(doc.get("text", "")))
        blocks.append(f"[FILE] {file_path}\n{text[:30000]}")
    joined = "\n\n".join(blocks).strip()
    return (
        "[BEGIN COMPRESSED CONTEXT]\n"
        f"{joined}\n"
        "[END COMPRESSED CONTEXT]"
    )