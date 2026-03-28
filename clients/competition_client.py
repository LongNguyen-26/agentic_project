import hashlib
import json
import random
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import httpx

from config import config
from models.api_schemas import SessionResponse, SubmissionResponse, TaskResponse
from core.logger import get_logger


logger = get_logger(__name__)

class APIClient:
    def __init__(self):
        self.base_url = (config.BASE_URL or "").rstrip('/')
        self.api_key = config.API_KEY or ""
        self.client = httpx.Client(timeout=config.HTTP_TIMEOUT_SECONDS)
        self.session_id: Optional[str] = None
        self.access_token: Optional[str] = None
        self._checkpoint_path = Path(config.STORAGE_ROOT) / "session_checkpoint.json"

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        retry_on_unauthorized: bool = False,
    ) -> httpx.Response:
        if not self.base_url:
            raise ValueError("COMPETITION_BASE_URL is required")
        if not self.api_key:
            raise ValueError("API_KEY is required")
        url = f"{self.base_url}{endpoint}"
        attempts = max(config.HTTP_MAX_RETRIES, 1)
        delay = max(config.HTTP_BACKOFF_SECONDS, 0.1)

        for attempt in range(1, attempts + 1):
            response: Optional[httpx.Response] = None
            try:
                response = self.client.request(
                    method,
                    url,
                    headers=headers,
                    json=json_data,
                    params=params,
                )
            except httpx.TransportError:
                logger.warning("[api] Transport error method=%s endpoint=%s attempt=%s/%s", method, endpoint, attempt, attempts, exc_info=True)
                if attempt >= attempts:
                    raise
                time.sleep((delay * attempt) + random.uniform(0, 0.25))
                continue

            if response.status_code == 401 and retry_on_unauthorized and self.session_id:
                logger.warning("[api] 401 received at %s; refreshing session", endpoint)
                self.create_session(self.session_id)
                if headers is not None and "Authorization" in headers and self.access_token:
                    headers["Authorization"] = f"Bearer {self.access_token}"
                if attempt >= attempts:
                    return response
                time.sleep((delay * attempt) + random.uniform(0, 0.25))
                continue

            if response.status_code in (429, 500, 502, 503, 504):
                logger.warning("[api] Retryable status=%s endpoint=%s attempt=%s/%s", response.status_code, endpoint, attempt, attempts)
                if attempt >= attempts:
                    return response
                time.sleep((delay * attempt) + random.uniform(0, 0.25))
                continue

            return response

        raise RuntimeError("Unexpected retry loop termination")

    def _persist_session_checkpoint(self):
        payload = {
            "session_id": self.session_id,
            "access_token": self.access_token,
        }
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_checkpoint(self) -> Optional[str]:
        if not self._checkpoint_path.exists():
            return None
        try:
            data = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
            self.session_id = data.get("session_id")
            self.access_token = data.get("access_token")
            return self.session_id
        except (json.JSONDecodeError, OSError):
            return None

    def create_session(self, session_id: Optional[str] = None) -> SessionResponse:
        headers = {"X-API-Key": self.api_key}
        data = {}
        if session_id:
            data["session_id"] = session_id

        response = self._request("POST", "/sessions", headers=headers, json_data=data)
        response.raise_for_status()
        session_resp = SessionResponse(**response.json())
        self.session_id = session_resp.session_id
        self.access_token = session_resp.access_token
        self._persist_session_checkpoint()
        logger.info("[auth] Session ready session_id=%s expires_in=%s", session_resp.session_id, session_resp.expires_in)
        return session_resp

    def authenticate(self) -> bool:
        """Compatibility facade for node layer; creates or refreshes a session."""
        try:
            self.ensure_session()
            return bool(self.session_id and self.access_token)
        except Exception:
            return False

    def ensure_session(self):
        if self.session_id and self.access_token:
            return
        existing = self.load_checkpoint()
        if existing:
            # Force refresh with the stored session id to keep session continuity.
            self.create_session(existing)
            return
        self.create_session()

    def get_next_task(self) -> Optional[TaskResponse]:
        self.ensure_session()

        headers = {"Authorization": f"Bearer {self.access_token}"}

        response = self._request("GET", "/tasks/next", headers=headers, retry_on_unauthorized=True)
        if response.status_code == 404:
            logger.info("[task] No more tasks available (404)")
            return None  # No more tasks
        response.raise_for_status()
        task = TaskResponse(**response.json())
        logger.info("[task] Fetched task_id=%s type=%s resources=%s", task.task_id, task.type, len(task.resources))
        return task

    def fetch_next_task(self) -> Optional[Dict[str, Any]]:
        """Return normalized task payload consumed by the outer graph."""
        task = self.get_next_task()
        if task is None:
            return None
        resources = [resource.model_dump(mode="json") for resource in task.resources]
        return {
            "id": task.task_id,
            "task_id": task.task_id,
            "type": task.type or "question-answering",
            "prompt_template": task.prompt_template,
            "resources": resources,
        }

    def download_file(self, token: str) -> bytes:
        self.ensure_session()
        headers = {"Authorization": f"Bearer {self.access_token}"}
        params = {"token": token}

        response = self._request("GET", "/download", headers=headers, params=params, retry_on_unauthorized=True)
        response.raise_for_status()
        return response.content

    def download_and_persist_resource(
        self,
        *,
        task_id: str,
        file_path: str,
        token: str,
    ) -> Dict[str, Any]:
        if not self.session_id:
            self.ensure_session()
        content = self.download_file(token)

        root = Path(config.STORAGE_ROOT) / (self.session_id or "unknown_session") / task_id / "raw"
        relative_path = Path(file_path)
        target = root / relative_path

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        except OSError as e:
            # Windows may fail on very deep/long paths; fallback to a flat hashed filename.
            safe_suffix = relative_path.suffix if relative_path.suffix else ".bin"
            safe_name = f"{hashlib.sha256(file_path.encode('utf-8')).hexdigest()}{safe_suffix}"
            fallback_dir = root / "_flat"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            target = fallback_dir / safe_name
            target.write_bytes(content)
            logger.warning(
                "[resource] Failed nested persist for %s (%s); saved to fallback path=%s",
                file_path,
                e,
                target,
            )
        logger.debug("[resource] Saved file_path=%s bytes=%s", file_path, len(content))

        checksum = hashlib.sha256(content).hexdigest()
        return {
            "file_path": file_path,
            "token": token,
            "local_path": str(target),
            "size": len(content),
            "sha256": checksum,
            "bytes": content,
        }

    def submit_task(self, task_id: str, answers: List[str], thought_log: str, used_tools: List[str]) -> SubmissionResponse:
        self.ensure_session()

        headers = {"Authorization": f"Bearer {self.access_token}"}
        data = {
            "session_id": self.session_id,
            "task_id": task_id,
            "answers": answers,
            "thought_log": thought_log,
            "used_tools": used_tools
        }

        response = self._request("POST", "/submissions", headers=headers, json_data=data, retry_on_unauthorized=True)
        if response.status_code == 409:
            # Already submitted for this task in the same session.
            logger.warning("[submit] Task already submitted task_id=%s", task_id)
            return SubmissionResponse(
                task_id=task_id,
                session_id=self.session_id or "",
                total_files=0,
                correct=0,
                score=0.0,
                details=[]
            )
        response.raise_for_status()
        submission = SubmissionResponse(**response.json())
        logger.info(
            "[submit] Accepted task_id=%s score=%s correct=%s/%s",
            submission.task_id,
            submission.score,
            submission.correct,
            submission.total_files,
        )
        return submission

    def submit_task_result(self, task_id: str, result: Dict[str, Any]) -> bool:
        """Compatibility facade for node layer result submission."""
        payload = result or {}
        answers = payload.get("answers", [])
        thought_log = payload.get("thought_log", "")
        used_tools = payload.get("used_tools", [])
        # self.submit_task(task_id=task_id, answers=answers, thought_log=thought_log, used_tools=used_tools)
        try:
            self.submit_task(task_id=task_id, answers=answers, thought_log=thought_log, used_tools=used_tools)
            return True
        except Exception as e:
            logger.error("[submit] Failed to submit task_result: %s", e)
            return False

    def close(self):
        self.client.close()