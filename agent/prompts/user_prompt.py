# prompts/user_prompt.py

def build_action_prompt(
    task_type: str,
    prompt_template: str,
    context: str,
    feedback: str = "",
    planning_hints: str = "",
) -> str:
    base_prompt = (
        f"Nhiệm vụ: {task_type}\n"
        f"Đề bài gốc:\n{prompt_template}\n\n"
    )

    if planning_hints:
        base_prompt += (
            "[CẢNH BÁO / HINTS TRƯỚC KHI GIẢI]\n"
            f"{planning_hints}\n\n"
        )

    base_prompt += (
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


def build_hints_extraction_prompt(prompt_template: str) -> str:
    return (
        "Bạn đang ở bước chuẩn bị trước khi giải task VPP.\n"
        "Chỉ đọc đề bài và trích xuất các rủi ro/cảnh báo cần lưu ý, chưa được giải bài.\n\n"
        "Tập trung vào các nhóm bẫy sau:\n"
        "1. Bẫy định dạng output (thứ tự trường, dấu phân tách, đơn vị, cách ghi rỗng).\n"
        "2. Placeholder bị ẩn dạng [tag_name] phải giữ nguyên verbatim.\n"
        "3. Từ ngữ dễ gây nhầm lẫn (đặc biệt thuật ngữ tiếng Nhật hoặc cụm đa nghĩa).\n"
        "4. Điều kiện đặc biệt có thể làm sai câu trả lời nếu bỏ sót.\n\n"
        "Yêu cầu đầu ra: liệt kê ngắn gọn các cảnh báo theo bullet, không đưa đáp án.\n\n"
        f"[PROMPT TEMPLATE]\n{prompt_template}"
    )


def build_file_summary_prompt(file_name: str, file_content: str) -> str:
    return (
        "Hãy tóm tắt tài liệu dưới đây để phục vụ cache ngữ cảnh.\n"
        "Yêu cầu:\n"
        "1. Tóm tắt 3-5 câu, ngắn gọn, nêu đúng nội dung chính của file.\n"
        "2. Liệt kê rõ các placeholder dạng [tag_name] xuất hiện trong file (nếu có), giữ nguyên từng chuỗi.\n"
        "3. Không suy diễn ngoài nội dung file.\n\n"
        f"[FILE NAME]\n{file_name}\n\n"
        f"[FILE CONTENT]\n{file_content}"
    )