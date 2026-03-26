import instructor
import json
import random
import time
from typing import Any, Dict, List, Optional, Type

from instructor.core.exceptions import IncompleteOutputException, InstructorRetryException
from openai import APIError, APITimeoutError, BadRequestError, OpenAI, RateLimitError
from pydantic import BaseModel, ValidationError

from config import config
from core.logger import get_logger
from models.llm_schemas import ActionPlanResponse, TaskClassification, VerificationResponse


logger = get_logger(__name__)

class LLMService:
    def __init__(self):
        self.client = instructor.from_openai(OpenAI(api_key=config.OPENAI_API_KEY))
        self.reasoning_model = config.MODEL_REASONING_ID

    def _build_evidence_block(self, retrieved_evidence: List[Dict[str, Any]]) -> str:
        max_items = max(config.EVIDENCE_MAX_ITEMS, 1)
        max_chars = max(config.EVIDENCE_MAX_CHARS_PER_ITEM, 200)
        compact_items = retrieved_evidence[:max_items]

        lines: List[str] = []
        for idx, item in enumerate(compact_items, start=1):
            metadata = item.get("metadata", {}) or {}
            file_hint = metadata.get("file_path") or metadata.get("category_name") or "unknown"
            content = (item.get("content") or "")[:max_chars]
            lines.append(f"[Evidence #{idx}] source={file_hint}\n{content}")
        return "\n\n".join(lines)

    def _chat_with_retries(
        self,
        *,
        response_model: Type[BaseModel],
        messages: List[Dict[str, str]],
        max_completion_tokens: int,
        temperature: float,
    ) -> BaseModel:
        attempts = max(config.LLM_MAX_RETRIES, 1)
        base_delay = max(config.LLM_RETRY_BASE_DELAY, 0.1)
        jitter = max(config.LLM_RETRY_JITTER, 0.0)
        last_exc: Optional[Exception] = None
        current_max_tokens = max(int(max_completion_tokens), 64)

        for attempt in range(1, attempts + 1):
            try:
                request_kwargs = {
                    "model": self.reasoning_model,
                    "response_model": response_model,
                    "messages": messages,
                    "max_completion_tokens": current_max_tokens,
                    "max_retries": 1,
                }
                # o-series reasoning models reject temperature.
                if not self.reasoning_model.lower().startswith("o"):
                    request_kwargs["temperature"] = temperature
                return self.client.chat.completions.create(**request_kwargs)
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
                token_limit_hit = (
                    "max_tokens length limit" in err_text
                    or "max_tokens or model output limit was reached" in err_text
                )
                if token_limit_hit and attempt < attempts:
                    current_max_tokens = min(current_max_tokens * 2, 8192)
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
                token_limit_hit = "max_tokens or model output limit was reached" in err_text

                if token_limit_hit and attempt < attempts:
                    current_max_tokens = min(current_max_tokens * 2, 8192)
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

    def classify_task_type(self, prompt_template: str) -> str:
        """Classify task type from Japanese/English prompt templates."""
        system_prompt = """You are a highly accurate task routing classifier.
Your job is to analyze a prompt (usually in Japanese) and classify it into exactly one of two task types:

1. "folder-organisation": The prompt asks the agent to sort, categorize, organize, or move files into specific folders/categories. 
   Keywords to look out for in Japanese: 仕分け (sorting), フォルダ (folder), 分類 (classification/categorization), 整理 (organizing), 振り分け (distribution).
2. "question-answering": The prompt asks a specific question, requests data extraction, or asks for a summary.
   Keywords to look out for in Japanese: 抽出 (extract), 教えて (tell me), 何ですか (what is), 答えて (answer).

Return ONLY a valid JSON object with a single key "task_type". The value MUST be either "folder-organisation" or "question-answering". Do not include any other keys or text."""

        user_prompt = f"Prompt Template to classify:\n{prompt_template}"

        try:
            response = self._chat_with_retries(
                response_model=TaskClassification,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=max(config.CLASSIFICATION_MAX_TOKENS, 256),
                temperature=1.0,
            )
            return response.task_type
        except Exception:
            logger.warning("[llm] Task classification failed; defaulting to question-answering", exc_info=True)
            return "question-answering"

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        max_completion_tokens: Optional[int] = None,
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
        )

    def generate_action_plan_and_answer(self, *, prompt_template: str, retrieved_evidence: list, task_type: str) -> dict:
        evidence_block = self._build_evidence_block(retrieved_evidence)

        system_prompt = (
            "You are an expert AI agent in a document-grounded simulation. "
            "Use only supplied evidence. Return JSON keys: "
            "answers (list[str]), thought_log (str), used_tools (list[str]), confidence (float 0-1)."
        )
        user_prompt = (
            f"Task type: {task_type}\n"
            f"Prompt template:\n{prompt_template}\n\n"
            f"Retrieved evidence:\n{evidence_block}\n\n"
            "Create the best possible answer with concise reasoning that references evidence."
        )

        try:
            response = self._chat_with_retries(
                response_model=ActionPlanResponse,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1.0,
                max_completion_tokens=config.LLM_MAX_OUTPUT_TOKENS,
            )
            return response.model_dump(mode="json")
        except ValidationError as e:
            logger.warning("[llm] Action response validation failed", exc_info=True)
            return ActionPlanResponse(thought_log=f"ValidationError: {e}").model_dump(mode="json")
        except Exception as e:
            logger.error("[llm] Action generation failed", exc_info=True)
            return ActionPlanResponse(thought_log=f"Error: {e}").model_dump()
    def verify_and_correct(
        self,
        *,
        prompt_template: str,
        candidate_answers: List[str],
        candidate_thought_log: str,
        retrieved_evidence: List[Dict[str, Any]],
    ) -> dict:
        evidence_block = self._build_evidence_block(retrieved_evidence)

        system_prompt = (
            "You are an expert AI agent in a document-grounded simulation. "
            "Use only supplied evidence to verify and correct the initial answer. Return JSON keys: "
            "answers (list[str]), thought_log (str), used_tools (list[str]), confidence (float 0-1), changed (bool)."
        )
        user_prompt = (
            f"Task type: question-answering\n"
            f"Prompt template:\n{prompt_template}\n\n"
            f"Initial answers:\n{json.dumps(candidate_answers, ensure_ascii=False)}\n\n"
            f"Initial thought_log:\n{candidate_thought_log}\n\n"
            f"Retrieved evidence:\n{evidence_block}\n\n"
            "Verify if the initial answer is correct. If not, provide a corrected answer with concise reasoning that references evidence."
        )

        try:
            response = self._chat_with_retries(
                response_model=VerificationResponse,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1.0,
                max_completion_tokens=config.VERIFICATION_MAX_OUTPUT_TOKENS,
            )
            return response.model_dump(mode="json")
        except ValidationError as e:
            logger.warning("[llm] Verification response validation failed", exc_info=True)
            return VerificationResponse(thought_log=f"ValidationError: {e}").model_dump(mode="json")
        except Exception as e:
            logger.error("[llm] Verification generation failed", exc_info=True)
            return VerificationResponse(thought_log=f"Error: {e}").model_dump()
        
    def generate_folder_response(self, *, prompt_template: str, file_contents: Dict[str, str]) -> dict:
        retrieved_evidence = [
            {
                "content": content,
                "metadata": {"file_path": path},
            }
            for path, content in file_contents.items()
        ]
        evidence_block = self._build_evidence_block(retrieved_evidence)

        system_prompt = (
            "You are an expert AI agent in a document-grounded simulation. "
            "Use only supplied evidence to generate folder organization instructions. Return JSON keys: "
            "answers (list[str]), thought_log (str), used_tools (list[str]), confidence (float 0-1)."
        )
        user_prompt = (
            f"Task type: folder-organisation\n"
            f"Prompt template:\n{prompt_template}\n\n"
            f"Retrieved evidence:\n{evidence_block}\n\n"
            "Generate precise instructions for how to organize files into folders based on the prompt and evidence."
        )

        try:
            response = self._chat_with_retries(
                response_model=ActionPlanResponse,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1.0,
                max_completion_tokens=config.LLM_MAX_OUTPUT_TOKENS,
            )
            return response.model_dump(mode="json")
        except ValidationError as e:
            logger.warning("[llm] Folder response validation failed", exc_info=True)
            return ActionPlanResponse(thought_log=f"ValidationError: {e}").model_dump(mode="json")
        except Exception as e:
            logger.error("[llm] Folder response generation failed", exc_info=True)
            return ActionPlanResponse(thought_log=f"Error: {e}").model_dump()


# Backward compatible alias.
LLMClient = LLMService