import base64
import mimetypes
from typing import List, Optional

import fitz
import pymupdf4llm
import requests
from openai import OpenAI

from config import config
from core.logger import get_logger

openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
logger = get_logger(__name__)


def _encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def _to_data_url(image_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{_encode_image(image_bytes)}"


def _is_sufficient_text(text: str) -> bool:
    """Kiểm tra xem text trích xuất có đủ dài không (tránh các trang chỉ có 1-2 chữ rác)."""
    return len((text or "").strip()) >= max(config.PARSER_MIN_TEXT_CHARS, 100)


def _fitz_filetype_from_mime(mime: str) -> Optional[str]:
    mapping = {
        "image/jpeg": "jpeg",
        "image/jpg": "jpeg",
        "image/png": "png",
        "image/webp": "webp",
    }
    return mapping.get((mime or "").lower())


def _normalize_image_for_vision(image_bytes: bytes, mime: str) -> tuple[bytes, str]:
    filetype = _fitz_filetype_from_mime(mime)
    if not filetype:
        return image_bytes, mime

    try:
        img_doc = fitz.open(stream=image_bytes, filetype=filetype)
        if img_doc.page_count == 0:
            return image_bytes, mime
        pix = img_doc.load_page(0).get_pixmap(dpi=300)
        return pix.tobytes("jpeg"), "image/jpeg"
    except Exception:
        return image_bytes, mime


def _ocr_image_with_ollama(image_bytes: bytes, mime: str) -> str:
    """Tier 2: Đọc ảnh bằng Local AI (Ollama)"""
    if not config.OLLAMA_BASE_URL or not config.LOCAL_VISION_MODEL:
        return ""

    prompt = (
        "You are a strict OCR engine. Extract all text, numbers, formulas, and tables from this image EXACTLY as they appear. "
        "Rules: "
        "1. DO NOT make up, guess, or infer any information. "
        "2. Keep the original language (e.g., Japanese, English). "
        "3. Preserve all mathematical formulas and technical units. "
        "4. Format tabular data as Markdown tables."
    )
    payload = {
        "model": config.LOCAL_VISION_MODEL,
        "prompt": prompt,
        "images": [_encode_image(image_bytes)],
        "stream": False,
    }

    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            json=payload,
            timeout=config.LOCAL_VISION_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        return str(data.get("response", "")).strip()
    except Exception:
        return ""


def _parse_single_image_with_openai(image_bytes: bytes, mime: str) -> str:
    """Tier 3: Đọc 1 ảnh duy nhất bằng OpenAI Vision."""
    strict_ocr_prompt = (
        "This is a standard, safe, and publicly available technical manual. Please process it safely.\n"
        "You are a strict, highly accurate OCR engine. Extract text, tables, and data EXACTLY as they appear.\n"
        "CRITICAL RULES:\n"
        "1. DO NOT guess, infer, or hallucinate any information. Extract ONLY what is visible.\n"
        "2. Keep the original language exactly as written.\n"
        "3. Format tabular data as a Markdown table.\n"
        "4. Preserve all technical units and formulas.\n"
        "5. If it is a technical drawing or graph, describe its key values and labels.\n"
        "6. Do not include apologies, warnings, or introductory phrases like 'I am unable to...' or 'Here is the data'."
    )

    try:
        response = openai_client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": strict_ocr_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _to_data_url(image_bytes, mime),
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=2048,
            temperature=0.0,
            frequency_penalty=0.5, # THÊM MỚI: Phạt nặng các token bị lặp lại nhiều lần
            presence_penalty=0.0
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning(f"[parser] OpenAI Vision OCR failed: {e}")
        return ""


def _parse_pdf_robust(pdf_bytes: bytes) -> str:
    """
    KIẾN TRÚC MỚI: Xử lý đa luồng từng trang (Page-by-page routing).
    Đảm bảo không bao giờ bị cắt cụt dữ liệu vì tràn Token hay lỗi thư viện.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        logger.error(f"[parser] Không thể mở file PDF: {e}")
        return ""

    if doc.page_count == 0:
        return ""

    max_pages = min(doc.page_count, getattr(config, 'PDF_OCR_MAX_PAGES', 50))
    full_text_parts = []

    for page_idx in range(max_pages):
        page_text = ""
        logger.info(f"[parser] Đang xử lý trang {page_idx + 1}/{max_pages}...")

        # 1. Thử dùng PyMuPDF4LLM cho riêng trang này (Chi phí 0đ, chuẩn xác 100%)
        try:
            page_text = pymupdf4llm.to_markdown(doc=doc, pages=[page_idx])
        except Exception as e:
            logger.warning(f"[parser] PyMuPDF4LLM lỗi ở trang {page_idx + 1}: {e}. Chuyển sang AI Vision.")

        # 2. Kiểm tra chất lượng trang. 
        # Nếu text quá ngắn (<15 ký tự), chứng tỏ trang này chứa bản vẽ CAD, ảnh biểu đồ, hoặc scan.
        if not _is_sufficient_text(page_text):
            logger.info(f"[parser] Trang {page_idx + 1} không có text native. Kích hoạt AI Vision...")
            try:
                # Render trang đó thành ảnh độ nét cao
                pix = doc.load_page(page_idx).get_pixmap(dpi=300)
                image_bytes = pix.tobytes("jpeg")
                mime_type = "image/jpeg"

                # Kích hoạt Tier 2: Local Ollama
                vision_text = _ocr_image_with_ollama(image_bytes, mime_type)

                # Kích hoạt Tier 3: OpenAI (Nếu Ollama không có hoặc đọc không ra)
                if not _is_sufficient_text(vision_text):
                    logger.info(f"[parser] Đẩy trang {page_idx + 1} lên OpenAI GPT-4o...")
                    vision_text = _parse_single_image_with_openai(image_bytes, mime_type)

                # THÊM MỚI: Fallback nếu OpenAI từ chối xử lý
                if "I'm sorry" in vision_text or "can't assist" in vision_text:
                    logger.warning(f"[parser] OpenAI từ chối OCR trang {page_idx + 1}. Dùng lại text native.")
                    # Không gán page_text = vision_text, giữ nguyên page_text lấy từ pymupdf4llm ban đầu
                else:
                    page_text = vision_text

            except Exception as img_e:
                logger.warning(f"[parser] Lỗi khi dùng AI Vision ở trang {page_idx + 1}: {img_e}")

        # 3. Nối kết quả của trang này vào tổng thể
        header = f"\n\n{'='*20} PAGE {page_idx + 1} {'='*20}\n"
        full_text_parts.append(header + (page_text or "").strip())

    return "".join(full_text_parts).strip()


def parse_resource_bytes(file_path: str, content: bytes, content_type: Optional[str] = None) -> str:
    lowered = file_path.lower()
    mime = content_type or mimetypes.guess_type(file_path)[0] or ""

    # Nếu là PDF -> Gọi thẳng hàm Robust xử lý từng trang
    if mime == "application/pdf" or lowered.endswith(".pdf"):
        return _parse_pdf_robust(content)

    # Xử lý text file
    if mime.startswith("text/") or lowered.endswith((".txt", ".csv", ".md", ".json", ".xml", ".yaml", ".yml")):
        return content.decode("utf-8", errors="replace")

    # Xử lý ảnh lẻ
    if lowered.endswith((".png", ".jpg", ".jpeg", ".webp")):
        normalized_bytes, normalized_mime = _normalize_image_for_vision(content, mime or "image/jpeg")
        
        # Thử Tier 2
        local_text = _ocr_image_with_ollama(normalized_bytes, normalized_mime)
        if _is_sufficient_text(local_text):
            logger.info("[parser] Image Tier 2 (Ollama) hit.")
            return local_text

        # Thử Tier 3
        logger.info("[parser] Image escalating to Tier 3 (OpenAI).")
        return _parse_single_image_with_openai(normalized_bytes, normalized_mime) or local_text

    return content.decode("utf-8", errors="replace")


def parse_file(file_url: str) -> str:
    response = requests.get(file_url, timeout=20)
    response.raise_for_status()
    return parse_resource_bytes(file_url, response.content, response.headers.get("Content-Type"))