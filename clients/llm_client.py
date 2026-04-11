import instructor
import json
import os
import random
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Type

from instructor.core.exceptions import IncompleteOutputException, InstructorRetryException
from langfuse.openai import OpenAI
from openai import APIError, APITimeoutError, BadRequestError, RateLimitError
from pydantic import BaseModel, ValidationError

from config import config
from core.logger import get_logger
from models.llm_schemas import (
    PlanningHintsResponse,
    TaskClassification,
)


logger = get_logger(__name__)


def _truncate_for_log(value: str, max_chars: int = 2000) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}... [truncated {len(value) - max_chars} chars]"

class LLMService:
    def __init__(self):
        # Keep Langfuse env wiring centralized in Settings while preserving runtime overrides.
        if config.LANGFUSE_SECRET_KEY:
            os.environ.setdefault("LANGFUSE_SECRET_KEY", config.LANGFUSE_SECRET_KEY)
        if config.LANGFUSE_PUBLIC_KEY:
            os.environ.setdefault("LANGFUSE_PUBLIC_KEY", config.LANGFUSE_PUBLIC_KEY)
        if config.LANGFUSE_HOST:
            os.environ.setdefault("LANGFUSE_HOST", config.LANGFUSE_HOST)
            os.environ.setdefault("LANGFUSE_BASE_URL", config.LANGFUSE_HOST)

        self.client = instructor.from_openai(OpenAI(api_key=config.OPENAI_API_KEY))
        self.reasoning_model = config.MODEL_REASONING_ID

    def _chat_with_retries(
        self,
        *,
        response_model: Type[BaseModel],
        messages: List[Dict[str, str]],
        max_completion_tokens: int,
        temperature: float,
        reasoning_effort: Optional[str] = None,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> BaseModel:
        attempts = max(config.LLM_MAX_RETRIES, 1)
        base_delay = max(config.LLM_RETRY_BASE_DELAY, 0.1)
        jitter = max(config.LLM_RETRY_JITTER, 0.0)
        retry_token_growth = max(config.LLM_RETRY_TOKEN_GROWTH_FACTOR, 1.0)
        last_exc: Optional[Exception] = None
        current_max_tokens = max(int(max_completion_tokens), 64)
        current_messages = deepcopy(messages)

        def _next_retry_tokens(current_tokens: int) -> int:
            retry_cap = max(int(config.LLM_RETRY_MAX_OUTPUT_TOKENS), current_tokens)
            proposed = int(max(current_tokens + 256, current_tokens * retry_token_growth))
            return min(proposed, retry_cap)

        def _is_context_overflow_error(err_text: str) -> bool:
            lowered = err_text.lower()
            return (
                "maximum context length" in lowered
                or "context_length_exceeded" in lowered
                or "string_above_max_length" in lowered
                or "prompt is too long" in lowered
            )

        def _trim_messages_for_retry(payload: List[Dict[str, str]]) -> List[Dict[str, str]]:
            """Trim oldest non-system messages while preserving system and latest user input."""
            if len(payload) <= 2:
                trimmed = deepcopy(payload)
                for msg in reversed(trimmed):
                    if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                        content = msg["content"]
                        new_len = max(int(len(content) * 0.8), 400)
                        if new_len < len(content):
                            msg["content"] = content[:new_len]
                        break
                return trimmed

            systems = [m for m in payload if m.get("role") == "system"]
            non_system = [m for m in payload if m.get("role") != "system"]

            latest_user_idx = -1
            for idx in range(len(non_system) - 1, -1, -1):
                if non_system[idx].get("role") == "user":
                    latest_user_idx = idx
                    break

            removable_indices: List[int] = []
            for idx, item in enumerate(non_system):
                if idx == latest_user_idx:
                    continue
                removable_indices.append(idx)

            if removable_indices:
                drop_idx = removable_indices[0]
                non_system.pop(drop_idx)

            return systems + non_system

        for attempt in range(1, attempts + 1):
            try:
                system_prompt = ""
                user_prompt = ""
                for msg in current_messages:
                    if msg.get("role") == "system" and not system_prompt:
                        system_prompt = str(msg.get("content") or "")
                    if msg.get("role") == "user":
                        user_prompt = str(msg.get("content") or "")

                logger.debug(
                    "[llm] Request model=%s response_model=%s attempt=%s/%s session_id=%s task_id=%s system_prompt=%s user_prompt=%s",
                    self.reasoning_model,
                    response_model.__name__,
                    attempt,
                    attempts,
                    session_id or "",
                    task_id or "",
                    _truncate_for_log(system_prompt),
                    _truncate_for_log(user_prompt),
                )

                request_kwargs = {
                    "model": self.reasoning_model,
                    "response_model": response_model,
                    "messages": current_messages,
                    "max_completion_tokens": current_max_tokens,
                    "max_retries": 1,
                }
                # o-series reasoning models reject temperature.
                if not self.reasoning_model.lower().startswith("o"):
                    request_kwargs["temperature"] = temperature
                elif reasoning_effort:
                    request_kwargs["reasoning_effort"] = reasoning_effort

                if session_id:
                    request_kwargs["session_id"] = session_id
                if task_id:
                    request_kwargs["tags"] = [f"task:{task_id}"]
                    request_kwargs["metadata"] = {"task_id": task_id}

                response = self.client.chat.completions.create(**request_kwargs)
                try:
                    response_dump = response.model_dump(mode="json")
                    logger.debug(
                        "[llm] Parsed response model=%s payload=%s",
                        response_model.__name__,
                        _truncate_for_log(json.dumps(response_dump, ensure_ascii=False)),
                    )
                except Exception:
                    logger.debug(
                        "[llm] Parsed response model=%s object=%s",
                        response_model.__name__,
                        _truncate_for_log(str(response)),
                    )
                return response
            except (ValidationError, RateLimitError, APITimeoutError, APIError, IncompleteOutputException) as exc:
                last_exc = exc
                logger.warning(
                    "[llm] Retryable error model=%s attempt=%s/%s error=%s",
                    self.reasoning_model,
                    attempt,
                    attempts,
                    exc.__class__.__name__,
                )
                if attempt >= attempts:
                    break
                sleep_for = (base_delay * (2 ** (attempt - 1))) + random.uniform(0.0, jitter)
                time.sleep(sleep_for)
            except InstructorRetryException as exc:
                # Instructor can wrap truncation as RetryError/InstructorRetryException.
                err_text = str(exc)
                context_limit_hit = _is_context_overflow_error(err_text)
                token_limit_hit = (
                    "max_tokens length limit" in err_text
                    or "max_tokens or model output limit was reached" in err_text
                )
                if context_limit_hit and attempt < attempts:
                    next_messages = _trim_messages_for_retry(current_messages)
                    if next_messages == current_messages:
                        last_exc = exc
                        break
                    current_messages = next_messages
                    logger.warning(
                        "[llm] Context overflow on model=%s; trimmed messages to %d entries attempt=%s/%s",
                        self.reasoning_model,
                        len(current_messages),
                        attempt,
                        attempts,
                    )
                    sleep_for = (base_delay * (2 ** (attempt - 1))) + random.uniform(0.0, jitter)
                    time.sleep(sleep_for)
                    continue
                if token_limit_hit and attempt < attempts:
                    next_max_tokens = _next_retry_tokens(current_max_tokens)
                    if next_max_tokens <= current_max_tokens:
                        logger.warning(
                            "[llm] Instructor token truncation hit retry ceiling on model=%s; max_completion_tokens=%s",
                            self.reasoning_model,
                            current_max_tokens,
                        )
                        last_exc = exc
                        break
                    current_max_tokens = next_max_tokens
                    logger.warning(
                        "[llm] Instructor token truncation on model=%s; retrying with max_completion_tokens=%s attempt=%s/%s",
                        self.reasoning_model,
                        current_max_tokens,
                        attempt,
                        attempts,
                    )
                    sleep_for = (base_delay * (2 ** (attempt - 1))) + random.uniform(0.0, jitter)
                    time.sleep(sleep_for)
                    continue
                last_exc = exc
                break
            except BadRequestError as exc:
                err_text = str(exc)
                context_limit_hit = _is_context_overflow_error(err_text)
                token_limit_hit = "max_tokens or model output limit was reached" in err_text

                if context_limit_hit and attempt < attempts:
                    next_messages = _trim_messages_for_retry(current_messages)
                    if next_messages == current_messages:
                        last_exc = exc
                        break
                    current_messages = next_messages
                    logger.warning(
                        "[llm] Context overflow on model=%s; trimmed messages to %d entries attempt=%s/%s",
                        self.reasoning_model,
                        len(current_messages),
                        attempt,
                        attempts,
                    )
                    sleep_for = (base_delay * (2 ** (attempt - 1))) + random.uniform(0.0, jitter)
                    time.sleep(sleep_for)
                    continue

                if token_limit_hit and attempt < attempts:
                    next_max_tokens = _next_retry_tokens(current_max_tokens)
                    if next_max_tokens <= current_max_tokens:
                        logger.warning(
                            "[llm] Token limit hit retry ceiling on model=%s; max_completion_tokens=%s",
                            self.reasoning_model,
                            current_max_tokens,
                        )
                        last_exc = exc
                        break
                    current_max_tokens = next_max_tokens
                    logger.warning(
                        "[llm] Token limit hit on model=%s; retrying with max_completion_tokens=%s attempt=%s/%s",
                        self.reasoning_model,
                        current_max_tokens,
                        attempt,
                        attempts,
                    )
                    sleep_for = (base_delay * (2 ** (attempt - 1))) + random.uniform(0.0, jitter)
                    time.sleep(sleep_for)
                    continue

                # Other bad requests are usually non-retryable (e.g., invalid payload/model).
                detail = None
                if getattr(exc, "response", None) is not None:
                    try:
                        detail = exc.response.text
                    except Exception:
                        detail = None
                logger.error(
                    "[llm] Non-retryable bad request on model=%s detail=%s",
                    self.reasoning_model,
                    detail or "<no response body>",
                    exc_info=True,
                )
                raise exc

        if last_exc:
            raise last_exc
        raise RuntimeError("LLM call failed without exception details")

    def classify_task_type(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> str:
        """Classify task type using caller-provided prompts."""
        try:
            response = self._chat_with_retries(
                response_model=TaskClassification,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=max(config.CLASSIFICATION_MAX_TOKENS, 256),
                temperature=0.0,
                session_id=session_id,
                task_id=task_id,
            )
            return response.task_type
        except Exception:
            logger.warning("[llm] Task classification failed; defaulting to question-answering", exc_info=True)
            return "question-answering"

    def extract_planning_hints(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> str:
        """Extract concise planning hints using caller-provided prompts."""
        try:
            response = self._chat_with_retries(
                response_model=PlanningHintsResponse,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=max(config.CLASSIFICATION_MAX_TOKENS * 4, 256),
                temperature=0.2,
                reasoning_effort="high",
                session_id=session_id,
                task_id=task_id,
            )
        except Exception:
            logger.warning("[llm] Planning hints extraction failed; continue without hints", exc_info=True)
            return ""

        parts: List[str] = []
        for idx, hint in enumerate(response.hints, start=1):
            cleaned = hint.strip()
            if cleaned:
                parts.append(f"{idx}. {cleaned}")
        caution = response.caution.strip()
        if caution:
            parts.append(f"Luu y tong quat: {caution}")
        return "\n".join(parts)

    def generate_action_response(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        max_completion_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> BaseModel:
        """Standardized action generation wrapper around structured LLM output."""
        return self.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            session_id=session_id,
            task_id=task_id,
        )

    def generate_verification_response(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        max_completion_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> BaseModel:
        """Standardized verification generation wrapper around structured LLM output."""
        return self.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            session_id=session_id,
            task_id=task_id,
        )

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        max_completion_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> BaseModel:
        """Generic typed structured output call for node-level usage."""
        return self._chat_with_retries(
            response_model=response_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=max_completion_tokens or config.LLM_MAX_OUTPUT_TOKENS,
            temperature=config.TEMPERATURE,
            reasoning_effort=reasoning_effort,
            session_id=session_id,
            task_id=task_id,
        )


# Backward compatible alias.
LLMClient = LLMService