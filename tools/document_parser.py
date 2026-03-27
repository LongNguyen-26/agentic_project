import base64
import mimetypes
from typing import List, Optional

import fitz
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
    """Return True when extracted text is long enough to skip expensive OCR tiers."""
    return len((text or "").strip()) >= max(config.PARSER_MIN_TEXT_CHARS, 1)


def _fitz_filetype_from_mime(mime: str) -> Optional[str]:
    mapping = {
        "image/jpeg": "jpeg",
        "image/jpg": "jpeg",
        "image/png": "png",
        "image/webp": "webp",
    }
    return mapping.get((mime or "").lower())


def _normalize_image_for_vision(image_bytes: bytes, mime: str) -> tuple[bytes, str]:
    """Convert input image to a smaller JPEG for better vision API compatibility."""
    filetype = _fitz_filetype_from_mime(mime)
    if not filetype:
        return image_bytes, mime

    try:
        img_doc = fitz.open(stream=image_bytes, filetype=filetype)
        if img_doc.page_count == 0:
            return image_bytes, mime
        pix = img_doc.load_page(0).get_pixmap(dpi=140)
        return pix.tobytes("jpeg"), "image/jpeg"
    except Exception:
        return image_bytes, mime


def _parse_pdf_text_fallback(pdf_bytes: bytes) -> str:
    """Tier 1: extract text layer from PDF with PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts: List[str] = []
    for page in doc:
        text = (page.get_text("text") or "").strip()
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _ocr_image_with_ollama(image_bytes: bytes, mime: str) -> str:
    """Tier 2 helper: OCR image bytes using local Ollama multimodal model."""
    prompt = (
        "Trich xuat day du noi dung van ban va thong tin ky thuat trong anh tai lieu xay dung. "
        "Giu nguyen don vi, so lieu, ma hieu va tieu de."
    )
    payload = {
        "model": config.LOCAL_VISION_MODEL,
        "prompt": prompt,
        "images": [_encode_image(image_bytes)],
        "stream": False,
    }
    # try:
    #     response = requests.post(
    #         f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/generate",
    #         json=payload,
    #         timeout=config.LOCAL_VISION_TIMEOUT_SECONDS,
    #     )
    #     response.raise_for_status()
    #     data = response.json()
    #     text = str(data.get("response", "")).strip()
    #     if text:
    #         logger.info("[parser] Tier 2 local OCR success chars=%d mime=%s", len(text), mime)
    #     return text
    # except Exception:
    #     logger.warning("[parser] Tier 2 local OCR failed", exc_info=True)
    #     return ""
    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            json=payload,
            timeout=config.LOCAL_VISION_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        text = str(data.get("response", "")).strip()
        if text:
            logger.info("[parser] Tier 2 local OCR success chars=%d mime=%s", len(text), mime)
        return text
    except requests.exceptions.ConnectionError:
        # Bắt riêng lỗi không gọi được local host (10061)
        logger.warning("[parser] Tier 2 local OCR failed: Không thể kết nối. Vui lòng kiểm tra xem Ollama đã được bật (run local) chưa.")
        return ""
    except requests.exceptions.HTTPError as e:
        # Bắt lỗi nếu gọi được nhưng model chưa được pull (thường trả về 404)
        logger.warning(f"[parser] Tier 2 local OCR failed: Chưa pull model '{config.LOCAL_VISION_MODEL}' hoặc lỗi từ Ollama. Chi tiết: {e}")
        return ""
    except Exception as e:
        # Các lỗi khác in ra thông báo ngắn gọn thay vì cả cục traceback
        logger.warning(f"[parser] Tier 2 local OCR failed: {str(e)}")
        return ""


def _parse_pdf_with_ollama(pdf_bytes: bytes) -> str:
    """Tier 2: render PDF pages and OCR locally with Ollama."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count == 0:
        return ""

    max_pages = min(doc.page_count, max(config.PDF_OCR_MAX_PAGES, 1))
    parts: List[str] = []
    for page_idx in range(max_pages):
        try:
            pix = doc.load_page(page_idx).get_pixmap(dpi=150)
            image_bytes = pix.tobytes("jpeg")
            text = _ocr_image_with_ollama(image_bytes, "image/jpeg")
            if text.strip():
                parts.append(f"[Page {page_idx + 1}]\n{text.strip()}")
        except Exception:
            logger.warning("[parser] Tier 2 PDF page OCR failed page=%s", page_idx + 1, exc_info=True)

    return "\n\n".join(parts).strip()


def _parse_pdf_with_openai(pdf_bytes: bytes) -> str:
    """Tier 3: fallback OCR with GPT-4o vision high detail."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count == 0:
        return ""

    content: List[dict] = [
        {
            "type": "text",
            "text": (
                "Extract all useful text and technical details from these construction document pages. "
                "Keep numbers, units, model codes, and field labels exactly."
            ),
        }
    ]
    max_pages = min(doc.page_count, max(config.PDF_OCR_MAX_PAGES, 1))
    for page_idx in range(max_pages):
        pix = doc.load_page(page_idx).get_pixmap(dpi=180)
        image_bytes = pix.tobytes("jpeg")
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _to_data_url(image_bytes, "image/jpeg"),
                    "detail": "high",
                },
            }
        )

    try:
        response = openai_client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[{"role": "user", "content": content}],
            max_tokens=4096,
        )
        text = (response.choices[0].message.content or "").strip()
        if max_pages < doc.page_count:
            text += f"\n\n[Note: Only first {max_pages} pages were processed]"
        return text
    except Exception:
        logger.warning("[parser] Tier 3 PDF OCR failed", exc_info=True)
        return ""


def _parse_image_with_openai(image_bytes: bytes, mime: str) -> str:
    """Tier 3 image fallback with GPT-4o vision high detail."""
    try:
        response = openai_client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract visible text and key details from this construction document image.",
                        },
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
        )
        return (response.choices[0].message.content or "").strip()
    except Exception:
        logger.warning("[parser] Tier 3 image OCR failed", exc_info=True)
        return ""


def parse_resource_bytes(file_path: str, content: bytes, content_type: Optional[str] = None) -> str:
    """Parse a resource with tiered strategy to optimize cost and robustness."""
    lowered = file_path.lower()
    mime = content_type or mimetypes.guess_type(file_path)[0] or ""

    if mime == "application/pdf" or lowered.endswith(".pdf"):
        text = _parse_pdf_text_fallback(content)
        if _is_sufficient_text(text):
            logger.info("[parser] PDF Tier 1 hit chars=%d file=%s", len(text), file_path)
            return text

        logger.info("[parser] PDF Tier 1 insufficient chars=%d file=%s", len(text), file_path)
        local_text = _parse_pdf_with_ollama(content)
        if _is_sufficient_text(local_text):
            logger.info("[parser] PDF Tier 2 hit chars=%d file=%s", len(local_text), file_path)
            return local_text

        logger.info("[parser] PDF escalating to Tier 3 file=%s", file_path)
        vision_text = _parse_pdf_with_openai(content)
        return vision_text or local_text or text

    if mime.startswith("text/") or lowered.endswith((".txt", ".csv", ".md", ".json", ".xml", ".yaml", ".yml")):
        return content.decode("utf-8", errors="replace")

    if lowered.endswith((".png", ".jpg", ".jpeg", ".webp")):
        normalized_bytes, normalized_mime = _normalize_image_for_vision(content, mime or "image/jpeg")
        local_text = _ocr_image_with_ollama(normalized_bytes, normalized_mime)
        if _is_sufficient_text(local_text):
            logger.info("[parser] Image Tier 2 hit chars=%d file=%s", len(local_text), file_path)
            return local_text

        logger.info("[parser] Image escalating to Tier 3 file=%s", file_path)
        vision_text = _parse_image_with_openai(normalized_bytes, normalized_mime)
        return vision_text or local_text

    return content.decode("utf-8", errors="replace")


def parse_file(file_url: str) -> str:
    response = requests.get(file_url, timeout=20)
    response.raise_for_status()
    return parse_resource_bytes(file_url, response.content, response.headers.get("Content-Type"))