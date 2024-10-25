"""This module contain functions for working with model."""

import numpy.typing as npt
import torch.cuda
from ultralytics import YOLO

from .image_utils import resize_with_pad


def predict_with_model(
    model: YOLO,
    image: npt.NDArray,
    device: str,
    resize_image: bool,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Predict with model on image and return bbox."""
    if not torch.cuda.is_available():
        device = "cpu"
        # print(f"Cuda is unavailable, cpu device is in use!")

    if resize_image:
        image = resize_with_pad(image, (640, 640))

    results = model.predict(image, device=device, verbose=False)

    annotated_img = results[0].plot()
    bboxes = results[0].boxes.xyxy.cpu().numpy()

    return annotated_img, bboxes
