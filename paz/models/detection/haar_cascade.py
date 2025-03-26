import cv2
from keras.utils import get_file
import paz
import numpy as np


def preprocess(image):
    image = paz.image.RGB_to_GRAY(image)
    image = paz.to_numpy(image)
    return image


def get_default_zero_detection():
    return []


def postprocess(boxes, class_arg):
    if len(boxes) == 0:
        detections = get_default_zero_detection()
    else:
        boxes = paz.boxes.xywh_to_xyxy(boxes)
        detections = paz.boxes.append_class(boxes, class_arg).astype(int)
    return detections


def draw(image, boxes, color, thickness):
    for box in boxes:
        image = paz.draw.box(image, paz.to_numpy(box), color, thickness)
    return image


def download(label):
    URL = (
        "https://raw.githubusercontent.com/opencv/opencv/"
        "master/data/haarcascades/"
    )
    filename = "haarcascade_" + label + ".xml"
    filepath = get_file(filename, URL + filename, cache_subdir="paz/models")
    model = cv2.CascadeClassifier(filepath)
    return model.detectMultiScale


def HaarCascadeDetector(label, scale, neighbors, class_arg, color, thickness):
    """Haar cascade detector.

    # Arguments
        label: String. Postfix openCV haarcascades XML name
            e.g. `eye`, `frontalface_alt2`, `fullbody`
            see references for all labels.
        class_arg: int. Class label argument.
        scale = float. Scale for image reduction
        neighbors: int. Minimum neighbors

    # Reference
        - [Haar
            Cascades](https://github.com/opencv/opencv/tree/master/data/haarcascades)
    """
    detect = download(label)

    def call(image):
        """Detects faces from an RGB image.

        # Arguments
            image: Array of shape ``(H, W, 3)``.

        # Returns
            Boxes ``(num_boxes, 4)``.
        """
        gray_image = preprocess(image)
        boxes = detect(gray_image, scale, neighbors)
        boxes = postprocess(boxes, class_arg)
        image = draw(image, boxes, color, thickness)
        return paz.NamedTuple("State", image=image, boxes=boxes)

    return call


def HaarCascadeFaceDetector(
    scale, neighbors, class_arg, color=paz.draw.GREEN, thickness=2
):
    return HaarCascadeDetector(
        "frontalface_default", scale, neighbors, class_arg, color, thickness
    )


def HaarCascadeEyeDetector(
    scale, neighbors, class_arg, color=paz.draw.GREEN, thickness=2
):
    return HaarCascadeDetector(
        "eye", scale, neighbors, class_arg, color, thickness
    )
