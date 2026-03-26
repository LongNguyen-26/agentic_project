# tools/context_manager.py

def format_full_context(raw_text: str) -> str:
    """
    Định dạng lại toàn bộ văn bản để đưa vào Prompt khi không dùng RAG.
    Có thể thêm logic cắt bỏ các ký tự thừa, chuẩn hóa khoảng trắng tại đây.
    """
    # Xóa bớt khoảng trắng thừa để tiết kiệm token
    cleaned_text = " ".join(raw_text.split())
    
    formatted = f"""
    [BẮT ĐẦU TÀI LIỆU GỐC]
    {cleaned_text}
    [KẾT THÚC TÀI LIỆU GỐC]
    """
    return formatted