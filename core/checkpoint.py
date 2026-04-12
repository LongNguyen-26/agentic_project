# core/checkpoint.py
import json
import hashlib
import os
from typing import Any, Dict, Optional, Tuple

from config import config
from core.logger import get_logger


logger = get_logger(__name__)

# Default checkpoint location under storage root.
DEFAULT_CHECKPOINT_DIR = os.path.join(os.getcwd(), config.STORAGE_ROOT)
DEFAULT_CHECKPOINT_FILE = os.path.join(DEFAULT_CHECKPOINT_DIR, "session_checkpoint.json")
PARSED_CACHE_DIR = os.path.join(DEFAULT_CHECKPOINT_DIR, "parsed_cache")

def _ensure_dir_exists(filepath: str) -> None:
    """Ensure destination directory exists before writing a file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def _load_raw_checkpoint(filepath: str = DEFAULT_CHECKPOINT_FILE) -> Dict[str, Any]:
    """Load checkpoint payload as a dict; return empty dict when absent/invalid."""
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        logger.warning("[checkpoint] Failed to parse checkpoint payload", exc_info=True)
    return {}

def save_checkpoint(
    session_id: str,
    access_token: str,
    filepath: str = DEFAULT_CHECKPOINT_FILE,
    extra: Optional[Dict[str, Any]] = None,
) -> bool:
    """Persist session_id and access_token into checkpoint JSON."""
    _ensure_dir_exists(filepath)
    try:
        data: Dict[str, Any] = _load_raw_checkpoint(filepath)
        data["session_id"] = session_id
        data["access_token"] = access_token
        if extra:
            data.update(extra)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logger.info("[checkpoint] Saved checkpoint at %s", filepath)
        return True
    except Exception as e:
        logger.warning("[checkpoint] Failed to save checkpoint: %s", e, exc_info=True)
        return False

def load_checkpoint(filepath: str = DEFAULT_CHECKPOINT_FILE) -> Tuple[Optional[str], Optional[str]]:
    """Load session_id and access_token from checkpoint JSON."""
    data = _load_raw_checkpoint(filepath)
    if not data:
        logger.debug("[checkpoint] No existing checkpoint at %s", filepath)
        return None, None

    try:
        session_id = data.get("session_id")
        access_token = data.get("access_token")
        
        if session_id and access_token:
            logger.info("[checkpoint] Restored session checkpoint %s...", session_id[:8])
            return session_id, access_token
        else:
            logger.warning("[checkpoint] Checkpoint file is missing required fields")
            return None, None
            
    except Exception as e:
        logger.warning("[checkpoint] Failed to read checkpoint: %s", e, exc_info=True)
        return None, None

def clear_checkpoint(filepath: str = DEFAULT_CHECKPOINT_FILE) -> None:
    """Delete checkpoint file when session expires or should be reset."""
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            logger.info("[checkpoint] Removed checkpoint at %s", filepath)
        except Exception as e:
            logger.warning("[checkpoint] Failed to remove checkpoint: %s", e, exc_info=True)


def _persist_session_checkpoint(session_id: str, access_token: str, filepath: str = DEFAULT_CHECKPOINT_FILE) -> bool:
    """Backward-compatible wrapper for legacy callers."""
    return save_checkpoint(session_id=session_id, access_token=access_token, filepath=filepath)


def save_file_summary_cache(cache_data: Dict[str, Any], filepath: str = DEFAULT_CHECKPOINT_FILE) -> bool:
    """Persist cross-task file-summary cache into the existing session checkpoint."""
    data = _load_raw_checkpoint(filepath)
    data["file_summary_cache"] = cache_data
    _ensure_dir_exists(filepath)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logger.debug("[checkpoint] Persisted file summary cache (%d keys)", len(cache_data))
        return True
    except Exception as e:
        logger.warning("[checkpoint] Failed to persist file summary cache: %s", e, exc_info=True)
        return False


def load_file_summary_cache(filepath: str = DEFAULT_CHECKPOINT_FILE) -> Dict[str, Any]:
    """Load cross-task file-summary cache from checkpoint payload."""
    data = _load_raw_checkpoint(filepath)
    cached = data.get("file_summary_cache", {})
    return cached if isinstance(cached, dict) else {}


def _get_safe_filename(file_path: str) -> str:
    """Create a filesystem-safe cache filename from the source file path."""
    safe_hash = hashlib.sha256(file_path.encode("utf-8")).hexdigest()
    return f"{safe_hash}.txt"


def load_parsed_text_cache(file_path: str) -> Optional[str]:
    """Load parsed text from disk cache using file_path as the lookup key."""
    safe_name = _get_safe_filename(file_path)
    cache_file = os.path.join(PARSED_CACHE_DIR, safe_name)

    if not os.path.exists(cache_file):
        return None

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.warning("[cache] Failed to read parsed text cache %s: %s", cache_file, e, exc_info=True)
        return None


def save_parsed_text_cache(file_path: str, parsed_text: str) -> bool:
    """Save parsed text to disk cache using a file_path-derived key."""
    os.makedirs(PARSED_CACHE_DIR, exist_ok=True)
    safe_name = _get_safe_filename(file_path)
    cache_file = os.path.join(PARSED_CACHE_DIR, safe_name)

    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(parsed_text)
        logger.debug("[cache] Saved parsed text cache for %s", file_path)
        return True
    except Exception as e:
        logger.warning("[cache] Failed to write parsed text cache %s: %s", cache_file, e, exc_info=True)
        return False