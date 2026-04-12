# core/exceptions.py

class AgentBaseException(Exception):
    """Base exception class for the entire agent runtime."""
    pass

# ==========================================
# Competition API related errors (Outer Loop)
# ==========================================
class CompetitionAPIError(AgentBaseException):
    """Generic error while calling competition APIs."""
    pass

class AuthenticationError(CompetitionAPIError):
    """Authentication failed (invalid key or expired token)."""
    pass

class RateLimitExceededError(CompetitionAPIError):
    """HTTP 429: requests were sent too quickly."""
    pass

class NoMoreTasksError(CompetitionAPIError):
    """No more tasks are available from the competition server."""
    pass

# ==========================================
# Task processing related errors (Inner Loop)
# ==========================================
class DocumentParseError(AgentBaseException):
    """Failed to read or parse PDF/image resources."""
    pass

class VectorDBError(AgentBaseException):
    """Failed to initialize or query RAG components (FAISS/OpenAI)."""
    pass

class LLMGenerationError(AgentBaseException):
    """LLM generation failed in OpenAI or Instructor pipeline."""
    pass

class VerificationFailedError(AgentBaseException):
    """Task failed self-correction validation after maximum retries."""
    pass