"""File for test api."""

import json
import numpy as np
from PIL import Image

from main import app
from car_detection import DEMO_DATA_DIR
from car_detection.image_utils import image_to_base64


def api_test():
    """Tests api work."""
    lst_images_dir = list(DEMO_DATA_DIR.rglob("*.[jpg][png][jpeg]"))

    lst_img_base64 = []
    for img_fpath in lst_images_dir:
        image = Image.open(img_fpath)
        img_base64 = image_to_base64(image, format=image.format)
        lst_img_base64.append(img_base64)

    data = json.dumps(lst_img_base64)

    with app.test_client() as client:
        response = client.post(
            "/api/pred",
            data=data,
            content_type="application/json"
        )

        assert response.status_code == 200, "Response status code is not 200!"
        response_data = response.get_json()

        for result in response_data:
            np_image = np.array(result["pred_image"])
            np_bboxes = np.array(result["bboxes"])
            # pil_image = Image.fromarray(np_image.astype("uint8"), "RGB")

            print("Image:", np_image.shape)
            print(np_image)
            print("Bounding boxes:", np_bboxes)


if __name__ == '__main__':
    api_test()
