# prompts/user_prompt.py

def build_action_prompt(task_type: str, prompt_template: str, context: str, feedback: str = "") -> str:
    base_prompt = (
        f"Nhiệm vụ: {task_type}\n"
        f"Đề bài gốc:\n{prompt_template}\n\n"
        f"[DỮ LIỆU CUNG CẤP]\n{context}\n\n"
    )

    if feedback:
        base_prompt += f"⚠️ LƯU Ý TỪ LẦN THỬ TRƯỚC (BẠN ĐÃ LÀM SAI):\n{feedback}\nHãy khắc phục lỗi này trong lần trả lời này.\n"

    base_prompt += "\nHãy phân tích dữ liệu và trả lời chuẩn xác theo schema bắt buộc."
    return base_prompt

def build_verification_prompt(prompt_template: str, draft_answer, context: str) -> str:
    return (
        f"[ĐỀ BÀI GỐC]\n{prompt_template}\n\n"
        f"[NGỮ CẢNH GỐC]\n{context}\n\n"
        f"[CÂU TRẢ LỜI CẦN KIỂM DUYỆT]\n{draft_answer}\n\n"
        "Hãy đánh giá chính xác, sửa nếu cần, và xuất đúng VerificationResponse."
    )