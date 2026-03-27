# config.py
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
	BASE_URL: str = os.getenv("COMPETITION_BASE_URL", "http://localhost:8000/v1")
	API_KEY: str = os.getenv("API_KEY", "")

	OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
	OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
	MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o")
	MODEL_REASONING_ID: str = os.getenv("REASONING_MODEL", "o3-mini")
	# MODEL_REASONING_ID: str = os.getenv("REASONING_MODEL", "gpt-5.3-chat-latest")
	LOCAL_VISION_MODEL: str = os.getenv("LOCAL_VISION_MODEL", "qwen2.5vl:7b")
	OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
	TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))

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
	# CLASSIFICATION_MAX_TOKENS: int = int(os.getenv("CLASSIFICATION_MAX_TOKENS", "64"))
	CLASSIFICATION_MAX_TOKENS: int = int(os.getenv("CLASSIFICATION_MAX_TOKENS", "512"))
	LLM_MAX_OUTPUT_TOKENS: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1200"))
	VERIFICATION_MAX_OUTPUT_TOKENS: int = int(os.getenv("VERIFICATION_MAX_OUTPUT_TOKENS", "1200"))
	PARSER_MIN_TEXT_CHARS: int = int(os.getenv("PARSER_MIN_TEXT_CHARS", "80"))
	PDF_OCR_MAX_PAGES: int = int(os.getenv("PDF_OCR_MAX_PAGES", "20"))
	LOCAL_VISION_TIMEOUT_SECONDS: float = float(os.getenv("LOCAL_VISION_TIMEOUT_SECONDS", "90"))

	QA_RAG_DOC_THRESHOLD: int = int(os.getenv("QA_RAG_DOC_THRESHOLD", "5"))
	RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "500"))
	RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
	RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "8"))
	EVIDENCE_MAX_ITEMS: int = int(os.getenv("EVIDENCE_MAX_ITEMS", "5"))
	EVIDENCE_MAX_CHARS_PER_ITEM: int = int(os.getenv("EVIDENCE_MAX_CHARS_PER_ITEM", "1200"))

	MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
	VERIFIER_MIN_CONFIDENCE: float = float(os.getenv("VERIFIER_MIN_CONFIDENCE", "0.60"))


config = Settings()
MAX_RETRIES = config.MAX_RETRIES