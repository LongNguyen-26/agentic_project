# core/exceptions.py

class AgentBaseException(Exception):
    """Lớp Exception cơ sở cho toàn bộ Agent."""
    pass

# ==========================================
# Nhóm lỗi liên quan đến API Cuộc thi (Outer Loop)
# ==========================================
class CompetitionAPIError(AgentBaseException):
    """Lỗi chung khi gọi API BTC."""
    pass

class AuthenticationError(CompetitionAPIError):
    """Lỗi xác thực (sai API Key, Token hết hạn)."""
    pass

class RateLimitExceededError(CompetitionAPIError):
    """Lỗi 429: Gửi request quá nhanh."""
    pass

class NoMoreTasksError(CompetitionAPIError):
    """BTC không còn task nào để cấp phát."""
    pass

# ==========================================
# Nhóm lỗi liên quan đến Xử lý Task (Inner Loop)
# ==========================================
class DocumentParseError(AgentBaseException):
    """Lỗi không thể đọc hoặc parse file PDF/Ảnh."""
    pass

class VectorDBError(AgentBaseException):
    """Lỗi khi khởi tạo hoặc query RAG (FAISS/Cohere)."""
    pass

class LLMGenerationError(AgentBaseException):
    """Lỗi từ phía OpenAI hoặc Instructor khi sinh kết quả."""
    pass

class VerificationFailedError(AgentBaseException):
    """Task không vượt qua được bài test self-correction sau n lần thử."""
    pass