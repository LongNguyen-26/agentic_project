import base64
import hashlib
import mimetypes
import os
import re
from typing import List, Optional
import concurrent.futures

import fitz
import pymupdf4llm
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
    """Post-processing cleanup for parser-specific image marker tags."""
    # Remove "intentionally omitted" image markers.
    text = re.sub(r'\*\*==> picture \[.*?\] intentionally omitted <==\*\*\n?', '', text)
    # Remove image block boundary tags.
    text = re.sub(r'\*\*----- (Start|End) of picture text -----\*\*(<br>)?\n?', '', text)
    return text.strip()


def _parse_single_image_with_openai(image_bytes: bytes, mime: str) -> str:
    """Tier 3: OCR a single image using OpenAI Vision."""
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


def _process_pdf_page(pdf_bytes: bytes, page_idx: int) -> tuple[int, str]:
    """Process one PDF page in isolation for concurrent parsing."""
    try:
        # Each worker opens a separate fitz document to avoid thread-safety issues.
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[page_idx]

        try:
            page_text = pymupdf4llm.to_markdown(doc=doc, pages=[page_idx])
        except Exception as e:
            logger.warning(f"[parser] PyMuPDF4LLM failed on page {page_idx + 1}: {e}.")
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
                logger.warning(f"[parser] Failed to cache image on page {page_idx + 1}: {exc}")

        if placeholders:
            page_text = (page_text or "").strip() + "\n\n" + "\n".join(placeholders)

        header = f"\n\n{'='*20} PAGE {page_idx + 1} {'='*20}\n"
        result_text = header + (page_text or "").strip()
        doc.close()
        return page_idx, result_text

    except Exception as e:
        logger.error(f"[parser] Page processing failed for page {page_idx + 1}: {e}")
        return page_idx, ""


def _parse_pdf_robust(pdf_bytes: bytes) -> str:
    """
    Parse PDF pages concurrently and cache image placeholders.
    """
    try:
        doc_temp = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = doc_temp.page_count
        doc_temp.close()
    except Exception as e:
        logger.error(f"[parser] Unable to open PDF stream: {e}")
        return ""

    if total_pages == 0:
        return ""

    max_pages = min(total_pages, getattr(config, 'PDF_OCR_MAX_PAGES', 50))
    full_text_parts = [""] * max_pages
    max_workers = min(10, max_pages)

    logger.info(f"[parser] Starting PDF parse for {max_pages} pages with {max_workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_page = {
            executor.submit(_process_pdf_page, pdf_bytes, page_idx): page_idx
            for page_idx in range(max_pages)
        }
        for future in concurrent.futures.as_completed(future_to_page):
            try:
                page_idx, text = future.result()
                full_text_parts[page_idx] = text
                logger.info(f"[parser] Finished rendering page {page_idx + 1}/{max_pages}.")
            except Exception as exc:
                logger.error(f"[parser] Worker raised an exception: {exc}")

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

        logger.info("[parser] Sending image to OpenAI Vision OCR.")
        openai_text = _parse_single_image_with_openai(normalized_bytes, normalized_mime)
        return _clean_parsed_text(openai_text)

    return content.decode("utf-8", errors="replace")
