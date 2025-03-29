from collections import namedtuple
from keras.utils import get_file
import cv2
import paz
import jax.numpy as jp


def download(label):
    URL = (
        "https://raw.githubusercontent.com/opencv/opencv/"
        "master/data/haarcascades/"
    )
    filename = "haarcascade_" + label + ".xml"
    filepath = get_file(filename, URL + filename, cache_subdir="paz/models")
    model = cv2.CascadeClassifier(filepath)
    return model.detectMultiScale


def get_empty_boxes():
    return jp.full((1, 5), -1)


def preprocess(image):
    return paz.image.RGB_to_GRAY(image)


def postprocess(boxes, class_arg):
    boxes = paz.boxes.xywh_to_xyxy(boxes)
    boxes = paz.boxes.append_class(boxes, class_arg).astype(int)
    return boxes


def HaarCascadeDetector(label, scale, neighbors, class_arg, draw=None):
    """Haar cascade detector."""
    detect = download(label)

    def call(RGB_image):
        gray_image = preprocess(RGB_image)
        boxes = detect(paz.to_numpy(gray_image), scale, neighbors)
        if len(boxes) == 0:
            boxes = get_empty_boxes()
        else:
            boxes = postprocess(boxes, class_arg)
        return paz.message.Detections(RGB_image, boxes)

    def call_and_draw(RGB_image):
        boxes = call(RGB_image).boxes
        return paz.message.Detections(draw(RGB_image, boxes), boxes)

    return call if draw is None else call_and_draw


def HaarCascadeFrontalFaceDetector(
    scale=1.3,
    neighbors=5,
    class_arg=0,
    draw=paz.lock(paz.draw.boxes, paz.draw.GREEN, 2),
):
    return HaarCascadeDetector(
        "frontalface_default", scale, neighbors, class_arg, draw
    )


def HaarCascadeEyeDetector(
    scale=1.3,
    neighbors=5,
    class_arg=0,
    draw=paz.lock(paz.draw.boxes, paz.draw.GREEN, 2),
):
    return HaarCascadeDetector("eye", scale, neighbors, class_arg, draw)
