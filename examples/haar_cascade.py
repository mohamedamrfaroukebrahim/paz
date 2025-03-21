import cv2
from keras.utils import get_file
import jax.numpy as jp
import paz


def preprocess(image):
    return paz.image.RGB_to_GRAY(image)


def default_no_boxes():
    return []


def postprocess(boxes, class_arg):
    if len(boxes) == 0:
        boxes = default_no_boxes()
    else:
        boxes = paz.boxes.xywh_to_xyxy(boxes)
        boxes = paz.boxes.append_class(boxes, class_arg)
        boxes = paz.cast(boxes, int)
    return boxes


def download(label):
    URL = (
        "https://raw.githubusercontent.com/opencv/opencv/"
        "master/data/haarcascades/"
    )
    filename = "haarcascade_" + label + ".xml"
    filepath = get_file(filename, URL + filename, cache_subdir="paz/models")
    model = cv2.CascadeClassifier(filepath)
    return model.detectMultiScale


def HaarCascadeDetector(label, scale, neighbors, class_arg=0):
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
        image = preprocess(image)
        boxes = detect(image, scale, neighbors)
        boxes = postprocess(boxes)

    return call
