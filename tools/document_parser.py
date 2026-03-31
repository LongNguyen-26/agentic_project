import base64
import mimetypes
import re
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def _parse_pdf_robust(pdf_bytes: bytes) -> str:
    """
    KIẾN TRÚC MỚI: Xử lý đa luồng từng trang (Page-by-page routing) với mô hình Crop-and-Reason.
    Sử dụng ThreadPoolExecutor để gọi OpenAI song song cho tất cả các ảnh hợp lệ trong PDF.
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
    
    # Gom tất cả các task xử lý ảnh trên mọi trang
    image_tasks = []
    # Dictionary lưu trữ kết quả phân tích ảnh theo index trang
    page_vision_results = {i: [] for i in range(max_pages)}

    for page_idx in range(max_pages):
        page = doc[page_idx]
        image_list = page.get_images(full=True)
        
        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            if not base_image:
                continue
                
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)
            
            # ==========================================
            # Image Gatekeeper (Filter Cải tiến)
            # ==========================================
            # Bỏ qua ảnh quá nhỏ (< 100px ở cả 2 chiều) hoặc nhiễu
            if width < 100 or height < 100:
                continue
                
            if height == 0: 
                continue
                
            aspect_ratio = width / height
            # Bỏ qua các đường thẳng kẻ vạch, border (tỷ lệ quá lệch)
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                continue
            
            image_bytes = base_image["image"]
            ext = base_image.get("ext", "jpeg")
            mime_type = f"image/{ext}"
            
            # Đưa vào danh sách chờ xử lý song song
            image_tasks.append({
                'page_idx': page_idx,
                'image_bytes': image_bytes,
                'mime_type': mime_type,
                'width': width,
                'height': height
            })

    # Xử lý song song bằng ThreadPoolExecutor (Giảm thời gian từ O(N) xuống O(1) dựa theo max_workers)
    if image_tasks:
        logger.info(f"[parser] Kích hoạt Multithreading Vision cho {len(image_tasks)} hình ảnh hợp lệ...")
        with ThreadPoolExecutor(max_workers=min(10, len(image_tasks))) as executor:
            future_to_task = {
                executor.submit(_parse_single_image_with_openai_for_charts, task['image_bytes'], task['mime_type']): task 
                for task in image_tasks
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    vision_desc = future.result()
                    if vision_desc:
                        page_vision_results[task['page_idx']].append(vision_desc)
                        logger.info(f"[parser] Hoàn thành phân tích ảnh trang {task['page_idx'] + 1} (w:{task['width']}, h:{task['height']}).")
                except Exception as exc:
                    logger.warning(f"[parser] Phân tích ảnh trang {task['page_idx'] + 1} thất bại: {exc}")

    # Lắp ráp lại văn bản cho từng trang
    for page_idx in range(max_pages):
        logger.info(f"[parser] Đang render nội dung trang {page_idx + 1}/{max_pages}...")
        try:
            page_text = pymupdf4llm.to_markdown(doc=doc, pages=[page_idx])
        except Exception as e:
            logger.warning(f"[parser] PyMuPDF4LLM lỗi ở trang {page_idx + 1}: {e}.")
            page_text = ""

        # Gắn kết quả phân tích biểu đồ/ảnh vào nội dung trang tương ứng
        for vision_desc in page_vision_results[page_idx]:
            page_text += f"\n\n> **[Mô tả Biểu đồ/Bản vẽ từ AI]**:\n> {vision_desc}\n"

        header = f"\n\n{'='*20} PAGE {page_idx + 1} {'='*20}\n"
        full_text_parts.append(header + (page_text or "").strip())

    final_text = "".join(full_text_parts).strip()
    
    # Gọi hàm Hậu xử lý Regex làm sạch tag trước khi trả về
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