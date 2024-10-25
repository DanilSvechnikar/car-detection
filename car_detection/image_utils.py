"""This module contain functions for working with images."""

import base64
import io
import cv2
import numpy.typing as npt
from PIL import Image


def resize_with_pad(frame: npt.NDArray, target_size: tuple[int, int]) -> npt.NDArray:
    """Resize image with black padding."""
    height, width = frame.shape[:2]
    target_h, target_w = target_size

    scale = min(target_w / width, target_h / height)
    new_w = int(width * scale)
    new_h = int(height * scale)

    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Calculate padding
    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    padded_frame = cv2.copyMakeBorder(
        resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color,
    )

    return padded_frame


def base64_to_image(base64_str: str):
    """Convert base64-str to image."""
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes))


def image_to_base64(image, format: str = None) -> str:
    """Convert image array to base64-str."""
    format_img = format if format is not None else image.format
    buffered = io.BytesIO()
    image.save(buffered, format=format_img)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
