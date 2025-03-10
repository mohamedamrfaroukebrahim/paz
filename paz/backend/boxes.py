import cv2
import jax.numpy as jp
import paz


def split(boxes):
    return jp.split(boxes, 4, axis=1)


def join(x_min, y_min, x_max, y_max):
    return jp.concatenate([x_min, y_min, x_max, y_max], axis=1)


def to_center_form(boxes):
    """Transform from corner coordinates to center coordinates.

    # Arguments
        boxes: Numpy array with shape `(num_boxes, 4)`.

    # Returns
        Numpy array with shape `(num_boxes, 4)`.
    """
    x_min, y_min, x_max, y_max = split(boxes)
    center_x = (x_max + x_min) / 2.0
    center_y = (y_max + y_min) / 2.0
    W = x_max - x_min
    H = y_max - y_min
    return jp.concatenate([center_x, center_y, W, H], axis=1)


def to_xywh(boxes):
    x_min, y_min = boxes[:, 0:1], boxes[:, 1:2]
    x_max, y_max = boxes[:, 2:3], boxes[:, 3:4]
    W = x_max - x_min
    H = y_max - y_min
    return jp.concatenate([x_min, y_min, W, H], axis=1)


def to_corner_form(boxes):
    """Transform from center coordinates to corner coordinates.

    # Arguments
        boxes: Numpy array with shape `(num_boxes, 4)`.

    # Returns
        Numpy array with shape `(num_boxes, 4)`.
    """
    center_x, center_y, W, H = split(boxes)
    x_min = center_x - (W / 2.0)
    x_max = center_x + (W / 2.0)
    y_min = center_y - (H / 2.0)
    y_max = center_y + (H / 2.0)
    return jp.concatenate([x_min, y_min, x_max, y_max], axis=1)


def pad(boxes, size, value=-1):
    """Pads boxes with given value.

    # Arguments
        boxes: Array `(num_boxes, 4)`.

    # Returns
        Padded boxes with shape `(size, 4)`.
    """
    num_boxes = len(boxes)
    if num_boxes > size:
        raise ValueError(f"Samples ({num_boxes}) exceeds pad ({size}).")
    padding = ((0, size - num_boxes), (0, 0))
    return jp.pad(boxes, padding, "constant", constant_values=value)


def pad_data(boxes, size, value=-1):
    """Pads list of boxes with a given.

    # Arguments
        boxes: List of size `(num_samples)` containing boxes as lists.

    # Returns
        boxes with shape `(num_samples, size, 4)`
    """
    padded_elements = []
    for sample in boxes:
        padded_element = pad(jp.array(sample), size, value)
        padded_elements.append(padded_element)
    return jp.array(padded_elements)


def flip_left_right(boxes, image_width):
    """Flips box coordinates from left-to-right and vice-versa.

    # Arguments
        boxes: Numpy array of shape `(num_boxes, 4)`.

    # Returns
        Numpy array of shape `(num_boxes, 4)`.
    """
    x_min, y_min, x_max, y_max = split(boxes)
    return join(x_max, y_min, x_min, y_max)


def from_selection(image, radius=5, color=(255, 0, 0), window_name="image"):
    points, boxes = [], []
    image = image.copy()

    def order_xyxy(point_A, point_B):
        (x1, y1) = point_A
        (x2, y2) = point_B
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def take_last_two_points(points):
        point_A = points[-1]
        point_B = points[-2]
        return point_A, point_B

    def on_double_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            paz.draw.circle(image, (x, y), radius, color)
            points.append((x, y))
            if len(points) % 2 == 0:
                point_A, point_B = take_last_two_points(points)
                box = order_xyxy(point_A, point_B)
                paz.draw.box(image, box, color, radius)
                boxes.append(box)

    def key_is_pressed(key="q", time=20):
        return cv2.waitKey(time) & 0xFF == ord(key)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_double_click)
    while True:
        paz.image.show(image, window_name, False)
        if key_is_pressed("q"):
            break
    cv2.destroyWindow(window_name)
    return jp.array(boxes)


def encode(matched, priors, variances=[0.1, 0.1, 0.2, 0.2]):
    """Encode the variances from the priorbox layers into the ground truth
    boxes we have matched (based on jaccard overlap) with the prior boxes.

    # Arguments
        matched: Numpy array of shape `(num_priors, 4)` with boxes in
            point-form.
        priors: Numpy array of shape `(num_priors, 4)` with boxes in
            center-form.
        variances: (list[float]) Variances of priorboxes

    # Returns
        encoded boxes: Numpy array of shape `(num_priors, 4)`.
    """
    boxes = matched[:, :4]
    boxes = to_center_form(boxes)
    center_difference_x = boxes[:, 0:1] - priors[:, 0:1]
    encoded_center_x = center_difference_x / priors[:, 2:3]
    center_difference_y = boxes[:, 1:2] - priors[:, 1:2]
    encoded_center_y = center_difference_y / priors[:, 3:4]
    encoded_center_x = encoded_center_x / variances[0]
    encoded_center_y = encoded_center_y / variances[1]
    encoded_W = jp.log((boxes[:, 2:3] / priors[:, 2:3]) + 1e-8)
    encoded_H = jp.log((boxes[:, 3:4] / priors[:, 3:4]) + 1e-8)
    encoded_W = encoded_W / variances[2]
    encoded_H = encoded_H / variances[3]
    encoded_boxes = [encoded_center_x, encoded_center_y, encoded_W, encoded_H]
    return jp.concatenate(encoded_boxes + [matched[:, 4:]], axis=1)


def decode(predictions, priors, variances=[0.1, 0.1, 0.2, 0.2]):
    """Decode default boxes into the ground truth boxes

    # Arguments
        loc: Numpy array of shape `(num_priors, 4)`.
        priors: Numpy array of shape `(num_priors, 4)`.
        variances: List of two floats. Variances of prior boxes.

    # Returns
        decoded boxes: Numpy array of shape `(num_priors, 4)`.
    """
    center_x = predictions[:, 0:1] * priors[:, 2:3] * variances[0]
    center_x = center_x + priors[:, 0:1]
    center_y = predictions[:, 1:2] * priors[:, 3:4] * variances[1]
    center_y = center_y + priors[:, 1:2]
    W = priors[:, 2:3] * jp.exp(predictions[:, 2:3] * variances[2])
    H = priors[:, 3:4] * jp.exp(predictions[:, 3:4] * variances[3])
    boxes = jp.concatenate([center_x, center_y, W, H], axis=1)
    boxes = to_corner_form(boxes)
    return jp.concatenate([boxes, predictions[:, 4:]], 1)
