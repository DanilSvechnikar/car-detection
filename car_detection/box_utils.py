"""This file contains functions related to bbox."""

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt


def read_bboxes(label_fpath: Path) -> tuple[list[list[float]], list[int]]:
    """Reads a bbox from a txt file."""
    bboxes = []
    class_labels = []
    with open(label_fpath, 'r') as file:
        for line in file:
            class_label, x_center, y_center, width, height = map(float, line.strip().split())
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(int(class_label))

    return bboxes, class_labels


def save_bboxes(
    label_file: Path,
    bboxes: list[np.float64],
    class_labels: list[int],
) -> None:
    """Saves the bbox to a txt file."""
    with open(label_file, "a") as file:
        for bbox, class_label in zip(bboxes, class_labels):
            file.write(f"{class_label} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")


def scale_bbox(
    original_bbox: npt.NDArray[np.float64],
    original_size: tuple[int, int],
    target_size: tuple[int, int],
) -> list[int]:
    """Scales the bbox."""
    # NOTE: The proportions of the image must be preserved
    orig_height, orig_width = original_size
    target_height, target_width = target_size

    scale = min(target_width / orig_width, target_height / orig_height)

    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)

    delta_w = (target_width - new_width) // 2
    delta_h = (target_height - new_height) // 2

    x1, y1, x2, y2 = original_bbox
    new_x1 = int(x1 * scale) + delta_w
    new_y1 = int(y1 * scale) + delta_h
    new_x2 = int(x2 * scale) + delta_w
    new_y2 = int(y2 * scale) + delta_h

    return [new_x1, new_y1, new_x2, new_y2]


def bbox2yolo(orig_bbox: list[int], image_size: tuple[int, int]) -> list[float]:
    """Convert bbox in xyxy to YOLO normalized xywh format."""
    img_height, img_width = image_size

    x1, y1, x2, y2 = orig_bbox

    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0

    width = x2 - x1
    height = y2 - y1

    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height

    return [x_center, y_center, width, height]


def yolo2bbox(bboxes: list[float]) -> tuple[float, float, float, float]:
    """Convert bounding boxes in YOLO format to xmin, ymin, xmax, ymax."""
    xmin, ymin = bboxes[0] - bboxes[2] / 2, bboxes[1] - bboxes[3] / 2
    xmax, ymax = bboxes[0] + bboxes[2] / 2, bboxes[1] + bboxes[3] / 2

    return xmin, ymin, xmax, ymax


def draw_boxes(
    image: np.ndarray,
    bboxes: list[list[float]],
    format_bbox: str = "yolo",
) -> tuple[np.ndarray, list[int]]:
    """Return image with drawed bbox."""
    box_areas = []

    if format_bbox == "coco":
        # coco has bboxes in xmin, ymin, width, height format
        # we need to add xmin and width to get xmax and...
        # ... ymin and height to get ymax
        for box_num, box in enumerate(bboxes):
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[0]) + int(box[2])
            ymax = int(box[1]) + int(box[3])

            width = int(box[2])
            height = int(box[3])

            cv2.rectangle(
                image,
                (xmin, ymin), (xmax, ymax),
                color=(0, 0, 255),
                thickness=2
            )
            box_areas.append(width * height)
    if format_bbox == "voc":
        for box_num, box in enumerate(bboxes):
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])

            width = xmax - xmin
            height = ymax - ymin

            cv2.rectangle(
                image,
                (xmin, ymin), (xmax, ymax),
                color=(0, 0, 255),
                thickness=2
            )
            box_areas.append(width * height)
    if format_bbox == "yolo":
        # need the image height and width to denormalize...
        # ... the bounding box coordinates
        h, w, _ = image.shape
        for box_num, box in enumerate(bboxes):
            x1, y1, x2, y2 = yolo2bbox(box)

            # denormalize the coordinates
            xmin = int(x1 * w)
            ymin = int(y1 * h)
            xmax = int(x2 * w)
            ymax = int(y2 * h)

            width = xmax - xmin
            height = ymax - ymin

            cv2.rectangle(
                image,
                (xmin, ymin), (xmax, ymax),
                color=(0, 0, 255),
                thickness=2
            )
            box_areas.append(width * height)

    return image, box_areas
