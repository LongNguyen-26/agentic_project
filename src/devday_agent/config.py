# config.py
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
	"""Centralized runtime configuration loaded from environment variables.

	Operationally important knobs:
	- QA_RAG_DOC_THRESHOLD:
	  Controls when QA tasks switch from full-context mode to hybrid retrieval mode.
	  Larger values reduce retrieval overhead for small tasks; smaller values favor retrieval
	  earlier when many resources are attached.
	- VERIFIER_MIN_CONFIDENCE:
	  Confidence gate for self-correction. Higher values enforce stricter acceptance but can
	  increase retries; lower values reduce retries but may accept weaker outputs.
	- MAX_RETRIES:
	  Hard limit for self-correction loops in verifiability routing.
	- LLM_MAX_OUTPUT_TOKENS / VERIFICATION_MAX_OUTPUT_TOKENS:
	  Upper bounds for model outputs to reduce truncation and overflow risk.
	"""
	# ── Competition server ─────────────────────────────────────────────────────
	BASE_URL: str = os.getenv("COMPETITION_BASE_URL", "")
	API_KEY: str = os.getenv("API_KEY", "")

	# ── OpenAI ─────────────────────────────────────────────────────────────────
	OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
	OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
	MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o")
	MODEL_REASONING_ID: str = os.getenv("REASONING_MODEL", "o3-mini")
	# MODEL_REASONING_ID: str = os.getenv("REASONING_MODEL", "gpt-5.3-chat-latest")

	# ── Runtime/Resilience ─────────────────────────────────────────────────────
	STORAGE_ROOT: str = os.getenv("STORAGE_ROOT", str(Path("storage")))

	LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
	LOG_FILE: str = os.getenv("LOG_FILE", str(Path("storage") / "agent.log"))
	LOG_MAX_BYTES: int = int(os.getenv("LOG_MAX_BYTES", "10485760"))
	LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))
	
	HTTP_TIMEOUT_SECONDS: float = float(os.getenv("HTTP_TIMEOUT_SECONDS", "40"))
	HTTP_MAX_RETRIES: int = int(os.getenv("HTTP_MAX_RETRIES", "4"))
	HTTP_BACKOFF_SECONDS: float = float(os.getenv("HTTP_BACKOFF_SECONDS", "1.0"))

	LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "4"))
	LLM_RETRY_BASE_DELAY: float = float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0"))
	LLM_RETRY_JITTER: float = float(os.getenv("LLM_RETRY_JITTER", "0.2"))

	# Stage-specific token budgets to avoid under-provisioning planning/summary calls.
	TASK_CLASSIFICATION_MAX_OUTPUT_TOKENS: int = int(os.getenv("TASK_CLASSIFICATION_MAX_OUTPUT_TOKENS", "256"))
	TASK_CLASSIFICATION_RETRY_MAX_OUTPUT_TOKENS: int = int(os.getenv("TASK_CLASSIFICATION_RETRY_MAX_OUTPUT_TOKENS", "1024"))
	PLANNING_HINTS_MAX_OUTPUT_TOKENS: int = int(os.getenv("PLANNING_HINTS_MAX_OUTPUT_TOKENS", "896"))
	PLANNING_HINTS_RETRY_MAX_OUTPUT_TOKENS: int = int(os.getenv("PLANNING_HINTS_RETRY_MAX_OUTPUT_TOKENS", "2400"))

	CLASSIFICATION_MAX_TOKENS: int = int(os.getenv("CLASSIFICATION_MAX_TOKENS", "64"))
	LLM_MAX_OUTPUT_TOKENS: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1800"))
	VERIFICATION_MAX_OUTPUT_TOKENS: int = int(os.getenv("VERIFICATION_MAX_OUTPUT_TOKENS", "1600"))
	QA_ACTION_MAX_OUTPUT_TOKENS: int = int(os.getenv("QA_ACTION_MAX_OUTPUT_TOKENS", os.getenv("LLM_MAX_OUTPUT_TOKENS", "1800")))
	QA_VERIFICATION_MAX_OUTPUT_TOKENS: int = int(os.getenv("QA_VERIFICATION_MAX_OUTPUT_TOKENS", os.getenv("VERIFICATION_MAX_OUTPUT_TOKENS", "1600")))
	QA_ACTION_RETRY_MAX_OUTPUT_TOKENS: int = int(os.getenv("QA_ACTION_RETRY_MAX_OUTPUT_TOKENS", "7200"))
	QA_VERIFICATION_RETRY_MAX_OUTPUT_TOKENS: int = int(os.getenv("QA_VERIFICATION_RETRY_MAX_OUTPUT_TOKENS", "5600"))

	# Sort tasks usually carry many file summaries; keep a high but bounded budget.
	SORT_ACTION_MAX_OUTPUT_TOKENS: int = int(os.getenv("SORT_ACTION_MAX_OUTPUT_TOKENS", "16000"))
	SORT_VERIFICATION_MAX_OUTPUT_TOKENS: int = int(os.getenv("SORT_VERIFICATION_MAX_OUTPUT_TOKENS", "10000"))
	SORT_ACTION_RETRY_MAX_OUTPUT_TOKENS: int = int(os.getenv("SORT_ACTION_RETRY_MAX_OUTPUT_TOKENS", "28000"))
	SORT_VERIFICATION_RETRY_MAX_OUTPUT_TOKENS: int = int(os.getenv("SORT_VERIFICATION_RETRY_MAX_OUTPUT_TOKENS", "16000"))
	# Retry policy for token-limit events.
	LLM_RETRY_MAX_OUTPUT_TOKENS: int = int(os.getenv("LLM_RETRY_MAX_OUTPUT_TOKENS", "36000"))
	LLM_RETRY_TOKEN_GROWTH_FACTOR: float = float(os.getenv("LLM_RETRY_TOKEN_GROWTH_FACTOR", "2.0"))

	# File summary compression knobs used by context manager.
	FILE_SUMMARY_SOURCE_MAX_CHARS: int = int(os.getenv("FILE_SUMMARY_SOURCE_MAX_CHARS", "80000"))
	FILE_SUMMARY_MAX_OUTPUT_TOKENS: int = int(os.getenv("FILE_SUMMARY_MAX_OUTPUT_TOKENS", "4096"))
	FILE_SUMMARY_RETRY_MAX_OUTPUT_TOKENS: int = int(os.getenv("FILE_SUMMARY_RETRY_MAX_OUTPUT_TOKENS", "8192"))

	# RAG prompt rendering controls to reduce repeated summary bloat.
	RAG_SUMMARY_MAX_CHARS: int = int(os.getenv("RAG_SUMMARY_MAX_CHARS", "280"))
	
	PDF_OCR_MAX_PAGES: int = int(os.getenv("PDF_OCR_MAX_PAGES", "30"))

	# ── QA Retrieval and Verification ───────────────────────────────────────────
	# Number of attached documents required before QA switches to RAG branch.
	QA_RAG_DOC_THRESHOLD: int = int(os.getenv("QA_RAG_DOC_THRESHOLD", "5"))

	RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
	RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
	RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "8"))
	RAG_RERANK_ENABLED: bool = os.getenv("RAG_RERANK_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
	RAG_RERANK_MODEL: str = os.getenv("RAG_RERANK_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
	RAG_RERANK_DEVICE: str = os.getenv("RAG_RERANK_DEVICE", "cpu")
	RAG_RERANK_PRE_TOP_K: int = int(os.getenv("RAG_RERANK_PRE_TOP_K", "20"))
	RAG_RERANK_BATCH_SIZE: int = int(os.getenv("RAG_RERANK_BATCH_SIZE", "16"))

	EVIDENCE_MAX_ITEMS: int = int(os.getenv("EVIDENCE_MAX_ITEMS", "5"))
	EVIDENCE_MAX_CHARS_PER_ITEM: int = int(os.getenv("EVIDENCE_MAX_CHARS_PER_ITEM", "1200"))

	# Maximum correction loops allowed by verification router.
	MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
	# Minimum confidence to pass verification without another retry.
	VERIFIER_MIN_CONFIDENCE: float = float(os.getenv("VERIFIER_MIN_CONFIDENCE", "0.60"))

	# Lightweight QA grounding gates to reduce hallucination risk.
	QA_GROUNDING_ENFORCED: bool = os.getenv("QA_GROUNDING_ENFORCED", "true").strip().lower() in {"1", "true", "yes", "on"}
	QA_GROUNDING_MIN_EVIDENCE_MARKERS: int = int(os.getenv("QA_GROUNDING_MIN_EVIDENCE_MARKERS", "1"))
	QA_ANSWER_LOG_MAX_CHARS: int = int(os.getenv("QA_ANSWER_LOG_MAX_CHARS", "12000"))


config = Settings()
MAX_RETRIES = config.MAX_RETRIES