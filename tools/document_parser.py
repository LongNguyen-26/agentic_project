import base64
import mimetypes
from typing import Optional

import fitz
import requests
from openai import BadRequestError, OpenAI

from config import config
from core.logger import get_logger
from models.llm_schemas import ExtractedDocumentData


openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
logger = get_logger(__name__)


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
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts = [page.get_text("text") for page in doc]
    return "\n".join(parts).strip()


def _parse_pdf_with_openai(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count == 0:
        return ""

    page = doc.load_page(0)

    for dpi in (180, 130, 96):
        try:
            pix = page.get_pixmap(dpi=dpi)
            image_bytes = pix.tobytes("jpeg")
            response = openai_client.beta.chat.completions.parse(
                model=config.MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "Extract the key text from the page and produce structured summary.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract relevant fields and summarize this page."},
                            {
                                "type": "image_url",
                                "image_url": {"url": _to_data_url(image_bytes, "image/jpeg")},
                            },
                        ],
                    },
                ],
                response_format=ExtractedDocumentData,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                continue
            output = parsed.summary.strip()
            if parsed.metadata:
                meta_lines = [f"{item.key}: {item.value}" for item in parsed.metadata if item.key or item.value]
                output = f"{output}\n\n[metadata]\n" + "\n".join(meta_lines)
            if output.strip():
                return output.strip()
        except BadRequestError as exc:
            logger.warning("[parser] PDF vision parse failed at dpi=%s: %s", dpi, exc)
            continue
        except Exception:
            logger.warning("[parser] PDF vision parse unexpected error at dpi=%s", dpi, exc_info=True)
            continue

    return _parse_pdf_text_fallback(pdf_bytes)


def parse_resource_bytes(file_path: str, content: bytes, content_type: Optional[str] = None) -> str:
    lowered = file_path.lower()
    mime = content_type or mimetypes.guess_type(file_path)[0] or ""

    if mime == "application/pdf" or lowered.endswith(".pdf"):
        return _parse_pdf_with_openai(content)

    if mime.startswith("text/") or lowered.endswith((".txt", ".csv", ".md", ".json", ".xml", ".yaml", ".yml")):
        return content.decode("utf-8", errors="replace")

    if lowered.endswith((".png", ".jpg", ".jpeg", ".webp")):
        normalized_bytes, normalized_mime = _normalize_image_for_vision(content, mime or "image/jpeg")
        try:
            response = openai_client.beta.chat.completions.parse(
                model=config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Extract text and short summary from this image."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract relevant text and summarize."},
                            {
                                "type": "image_url",
                                "image_url": {"url": _to_data_url(normalized_bytes, normalized_mime)},
                            },
                        ],
                    },
                ],
                response_format=ExtractedDocumentData,
            )
            parsed = response.choices[0].message.parsed
            return (parsed.summary if parsed else "").strip()
        except BadRequestError as exc:
            logger.warning("[parser] Image vision parse failed for %s: %s", file_path, exc)
            return ""
        except Exception:
            logger.warning("[parser] Image vision parse unexpected error for %s", file_path, exc_info=True)
            return ""

    return content.decode("utf-8", errors="replace")


def parse_file(file_url: str) -> str:
    response = requests.get(file_url, timeout=20)
    response.raise_for_status()
    return parse_resource_bytes(file_url, response.content, response.headers.get("Content-Type"))