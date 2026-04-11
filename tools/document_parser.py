import base64
import hashlib
import mimetypes
import os
import re
from typing import List, Optional
import concurrent.futures

import fitz
import pymupdf4llm
import requests
from openai import OpenAI

from config import config
from core.logger import get_logger

openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
logger = get_logger(__name__)
IMAGE_CACHE_DIR = os.path.join(os.getcwd(), config.STORAGE_ROOT, "image_cache")


def _save_image_to_cache(image_id: str, image_bytes: bytes) -> str:
    """Persist image bytes to disk cache for later tool-based analysis."""
    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
    image_path = os.path.join(IMAGE_CACHE_DIR, f"{image_id}.jpg")
    if not os.path.exists(image_path):
        with open(image_path, "wb") as cache_file:
            cache_file.write(image_bytes)
    return image_path


def _encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def _to_data_url(image_bytes: bytes, mime: str) -> str:
    return f"data:{mime};base64,{_encode_image(image_bytes)}"


def _is_sufficient_text(text: str) -> bool:
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


def _clean_parsed_text(text: str) -> str:
    """Hậu xử lý (Post-processing): Làm sạch các tag thừa bằng Regex."""
    # Xóa đánh dấu ảnh intentionally omitted
    text = re.sub(r'\*\*==> picture \[.*?\] intentionally omitted <==\*\*\n?', '', text)
    # Xóa các tag block của ảnh
    text = re.sub(r'\*\*----- (Start|End) of picture text -----\*\*(<br>)?\n?', '', text)
    return text.strip()


def _ocr_image_with_ollama(image_bytes: bytes, mime: str) -> str:
    """Tier 2: Đọc ảnh bằng Local AI (Ollama)"""
    if not config.OLLAMA_BASE_URL or not config.LOCAL_VISION_MODEL:
        return ""

    prompt = (
        "You are a strict OCR engine. Extract all text, numbers, formulas, and tables from this image EXACTLY as they appear. "
        "Rules: "
        "1. DO NOT make up, guess, or infer any information. "
        "2. Keep the original language (e.g., Japanese, Vietnamese, English). "
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
    """Tier 3: Đọc 1 ảnh duy nhất bằng OpenAI Vision (OCR thông thường)."""
    strict_ocr_prompt = (
        "This is a standard, safe, and publicly available technical manual. Please process it safely.\n"
        "You are a strict, highly accurate OCR engine. Extract text, tables, and data EXACTLY as they appear.\n"
        "CRITICAL RULES:\n"
        "1. DO NOT guess, infer, or hallucinate any information. Extract ONLY what is visible.\n"
        "2. MUST answer EXACTLY in the original language of the document (e.g., Japanese, Vietnamese). Do not translate.\n"
        "3. Format tabular data as a Markdown table.\n"
        "4. Preserve all technical units and formulas.\n"
        "5. Output ONLY the extracted Markdown data or Key Takeaways. Do NOT include introductory phrases like 'Here is the data...', 'The image shows...', etc."
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
            frequency_penalty=0.5,
            presence_penalty=0.0
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning(f"[parser] OpenAI Vision OCR failed: {e}")
        return ""


def _parse_single_image_with_openai_for_charts(image_bytes: bytes, mime: str) -> str:
    """Tier 3: Đọc và phân tích logic (Reasoning) cho Biểu đồ/Bản vẽ kỹ thuật bằng OpenAI Vision."""
    chart_reasoning_prompt = (
        "You are an expert technical data analyst. Deeply analyze this technical diagram, chart, or image.\n"
        "CRITICAL RULES:\n"
        "1. MUST answer EXACTLY in the original language of the document (e.g., Japanese, Vietnamese). Do not translate.\n"
        "2. Explain the core message, key values, trends, and components present in the image.\n"
        "3. Format your explanation clearly using Markdown, highlighting important data points.\n"
        "4. Output ONLY the extracted Markdown data or Key Takeaways. Do NOT include introductory phrases like 'Here is the data...', 'The image shows...', etc."
    )

    try:
        response = openai_client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": chart_reasoning_prompt},
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
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning(f"[parser] OpenAI Vision Chart Reasoning failed: {e}")
        return ""


def _process_pdf_page(pdf_bytes: bytes, page_idx: int) -> tuple[int, str]:
    """Xử lý một trang PDF độc lập để chạy đa luồng."""
    try:
        # Mỗi luồng mở một document riêng để tránh lỗi thread-safety của fitz.
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[page_idx]

        try:
            page_text = pymupdf4llm.to_markdown(doc=doc, pages=[page_idx])
        except Exception as e:
            logger.warning(f"[parser] PyMuPDF4LLM lỗi ở trang {page_idx + 1}: {e}.")
            page_text = ""

        placeholders: List[str] = []
        image_list = page.get_images(full=True)

        for img in image_list:
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                if width < 100 or height < 100 or height == 0:
                    continue

                aspect_ratio = width / height
                if aspect_ratio > 10 or aspect_ratio < 0.1:
                    continue

                image_bytes = base_image["image"]
                ext = base_image.get("ext", "jpeg")
                mime_type = f"image/{ext}"

                image_id = hashlib.sha256(image_bytes).hexdigest()[:16]
                normalized_bytes, _ = _normalize_image_for_vision(image_bytes, mime_type)
                _save_image_to_cache(image_id, normalized_bytes)

                placeholder = f"[IMAGE_PLACEHOLDER | ID: {image_id} | Size: {width}x{height}]"
                placeholders.append(placeholder)
            except Exception as exc:
                logger.warning(f"[parser] Không thể cache ảnh ở trang {page_idx + 1}: {exc}")

        if placeholders:
            page_text = (page_text or "").strip() + "\n\n" + "\n".join(placeholders)

        header = f"\n\n{'='*20} PAGE {page_idx + 1} {'='*20}\n"
        result_text = header + (page_text or "").strip()
        doc.close()
        return page_idx, result_text

    except Exception as e:
        logger.error(f"[parser] Lỗi xử lý trang {page_idx + 1}: {e}")
        return page_idx, ""


def _parse_pdf_robust(pdf_bytes: bytes) -> str:
    """
    Parse PDF đa luồng theo trang và cache image placeholders.
    """
    try:
        doc_temp = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = doc_temp.page_count
        doc_temp.close()
    except Exception as e:
        logger.error(f"[parser] Không thể mở file PDF: {e}")
        return ""

    if total_pages == 0:
        return ""

    max_pages = min(total_pages, getattr(config, 'PDF_OCR_MAX_PAGES', 50))
    full_text_parts = [""] * max_pages
    max_workers = min(10, max_pages)

    logger.info(f"[parser] Bắt đầu parse {max_pages} trang PDF với {max_workers} luồng...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_page = {
            executor.submit(_process_pdf_page, pdf_bytes, page_idx): page_idx
            for page_idx in range(max_pages)
        }
        for future in concurrent.futures.as_completed(future_to_page):
            try:
                page_idx, text = future.result()
                full_text_parts[page_idx] = text
                logger.info(f"[parser] Đã hoàn thành render nội dung trang {page_idx + 1}/{max_pages}.")
            except Exception as exc:
                logger.error(f"[parser] Trang sinh ra lỗi: {exc}")

    final_text = "".join(full_text_parts).strip()
    return _clean_parsed_text(final_text)


def parse_resource_bytes(file_path: str, content: bytes, content_type: Optional[str] = None) -> str:
    lowered = file_path.lower()
    mime = content_type or mimetypes.guess_type(file_path)[0] or ""

    if mime == "application/pdf" or lowered.endswith(".pdf"):
        return _parse_pdf_robust(content)

    if mime.startswith("text/") or lowered.endswith((".txt", ".csv", ".md", ".json", ".xml", ".yaml", ".yml")):
        return content.decode("utf-8", errors="replace")

    if lowered.endswith((".png", ".jpg", ".jpeg", ".webp")):
        normalized_bytes, normalized_mime = _normalize_image_for_vision(content, mime or "image/jpeg")
        
        local_text = _ocr_image_with_ollama(normalized_bytes, normalized_mime)
        if _is_sufficient_text(local_text):
            logger.info("[parser] Image Tier 2 (Ollama) hit.")
            return _clean_parsed_text(local_text)

        logger.info("[parser] Image escalating to Tier 3 (OpenAI).")
        openai_text = _parse_single_image_with_openai(normalized_bytes, normalized_mime)
        return _clean_parsed_text(openai_text or local_text)

    return content.decode("utf-8", errors="replace")


def parse_file(file_url: str) -> str:
    response = requests.get(file_url, timeout=20)
    response.raise_for_status()
    return parse_resource_bytes(file_url, response.content, response.headers.get("Content-Type"))