import base64
import os
from typing import List

from openai import OpenAI

from config import config
from core.logger import get_logger


logger = get_logger(__name__)
openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
IMAGE_CACHE_DIR = os.path.join(os.getcwd(), config.STORAGE_ROOT, "image_cache")


def _to_data_url(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def analyze_images_from_cache(image_ids: List[str], vision_prompt: str) -> str:
    """Analyze requested images from disk cache with OpenAI Vision.

    Args:
        image_ids: List of image hashes extracted from IMAGE_PLACEHOLDER tags.
        vision_prompt: Prompt passed to the vision model.

    Returns:
        Aggregated text observations for all requested images.
    """
    if not image_ids:
        return "[Vision Tool] No image IDs were requested."

    prompt = vision_prompt.strip() or "Describe the image in detail and extract all visible text."
    observations: List[str] = []

    for img_id in image_ids:
        image_path = os.path.join(IMAGE_CACHE_DIR, f"{img_id}.jpg")
        if not os.path.exists(image_path):
            msg = f"[Vision Tool][{img_id}] ERROR: image not found at {image_path}"
            logger.warning(msg)
            observations.append(msg)
            continue

        try:
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
        except Exception as exc:
            msg = f"[Vision Tool][{img_id}] ERROR: failed to read image file: {exc}"
            logger.warning(msg)
            observations.append(msg)
            continue

        try:
            response = openai_client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": _to_data_url(image_bytes),
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1200,
                temperature=0.0,
            )
            output = (response.choices[0].message.content or "").strip()
            if not output:
                output = "[Vision Tool] Empty model response."
            observations.append(f"[Vision Tool][{img_id}] {output}")
        except Exception as exc:
            msg = f"[Vision Tool][{img_id}] ERROR: OpenAI Vision API failed: {exc}"
            logger.warning(msg)
            observations.append(msg)

    return "\n\n".join(observations)
