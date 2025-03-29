from keras.utils import get_file
import cv2
import paz


def download(label):
    URL = (
        "https://raw.githubusercontent.com/opencv/opencv/"
        "master/data/haarcascades/"
    )
    filename = "haarcascade_" + label + ".xml"
    filepath = get_file(filename, URL + filename, cache_subdir="paz/models")
    model = cv2.CascadeClassifier(filepath)
    return model.detectMultiScale


def preprocess(image):
    image = paz.image.RGB_to_GRAY(image)
    image = paz.to_numpy(image)
    return image


def postprocess(boxes, class_arg):
    if len(boxes) == 0:
        detections = []
    else:
        boxes = paz.boxes.xywh_to_xyxy(boxes)
        detections = paz.boxes.append_class(boxes, class_arg).astype(int)
    return detections


def HaarCascadeDetector(
    label,
    scale,
    neighbors,
    class_arg,
    color,
    thickness,
):
    """Haar cascade detector.

    # Arguments
        label: String. Postfix openCV haarcascades XML name e.g `eye`, `face`.
            see references for all labels.
        scale = float. Scale for image reduction
        neighbors: int. Minimum neighbors
        class_arg: int. Class label argument.

    # Reference
        - [Haar Cascades](
        https://github.com/opencv/opencv/tree/master/data/haarcascades)
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
        image = paz.draw.boxes(image, boxes, color, thickness)
        return paz.NamedTuple("State", image=image, boxes=boxes)

    return call


def HaarCascadeFaceDetector(
    scale=1.3, neighbors=5, class_arg=0, color=paz.draw.GREEN, thickness=2
):
    return HaarCascadeDetector(
        "frontalface_default", scale, neighbors, class_arg, color, thickness
    )


def HaarCascadeEyeDetector(
    scale=1.3, neighbors=5, class_arg=0, color=paz.draw.GREEN, thickness=2
):
    return HaarCascadeDetector(
        "eye", scale, neighbors, class_arg, color, thickness
    )
