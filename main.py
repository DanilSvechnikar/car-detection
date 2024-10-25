"""Main app."""

import os
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

from car_detection import MODELS_DIR
from car_detection.models_utils import predict_with_model
from car_detection.image_utils import base64_to_image, image_to_base64

load_dotenv()
app = Flask(__name__)

page_fname = "index.html"
device = os.getenv("DEVICE")
resize_image = os.getenv("RESIZE_IMAGE").lower() in ("true", "1", "t")


@app.route("/")
def index():
    """Start page."""
    return render_template(page_fname)


@app.route("/upload", methods=["POST"])
def upload_image():
    """Loads images, makes predictions and displays them on the page."""
    files = request.files.getlist("files")
    if not files:
        return "Empty files!", 400

    pred_images = []
    pred_bboxes = []
    for file in files:
        image = Image.open(file.stream)
        np_image = np.array(image)

        annotated_img, bboxes = predict_with_model(
            MODEL, np_image, device, resize_image,
        )

        annotated_img = Image.fromarray(annotated_img)
        img_base64 = image_to_base64(annotated_img, image.format)

        pred_images.append(img_base64)
        pred_bboxes.append(bboxes)

    return render_template(page_fname, images=pred_images, bboxes=pred_bboxes, zip=zip)


@app.route("/api/pred", methods=["POST"])
def process_base64_image():
    """Takes an image, makes predictions and returns the results."""
    lst_img_base64 = request.get_json()

    if not lst_img_base64:
        return jsonify({"Error": "No images!"}), 400

    results = []
    for img_base64 in lst_img_base64:
        image = base64_to_image(img_base64)
        np_image = np.array(image)

        annotated_img, bboxes = predict_with_model(
            MODEL, np_image, device, resize_image,
        )

        annotated_img = annotated_img.tolist()
        bboxes = bboxes.tolist()
        results.append(
            {"pred_image": annotated_img, "bboxes": bboxes},
        )

    return jsonify(results)


def initialize_model():
    """Load model weights."""
    model_name = os.getenv("MODEL_NAME")
    model_path = MODELS_DIR / model_name

    model = YOLO(model_path)
    # print(f"Model {model_name} loaded successfully.")
    return model


MODEL = initialize_model()

if __name__ == "__main__":
    host = os.getenv("HOST")
    port = os.getenv("PORT")
    app.run(host=host, port=port)
