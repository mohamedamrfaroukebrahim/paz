import jax
import jax.numpy as jp
import cv2
import paz


def split(boxes, keepdims=True, axis=1):
    """Split boxes into x_min, y_min, x_max, y_max components."""
    coordinates = jp.split(boxes, 4, axis=axis)
    if not keepdims:
        coordinates = tuple(
            jp.squeeze(column, axis=-1) for column in coordinates
        )

    return coordinates


def merge(coordinate_0, coordinate_1, coordinate_2, coordinate_3):
    coordinates = [coordinate_0, coordinate_1, coordinate_2, coordinate_3]
    return jp.concatenate(coordinates, axis=1)


def compute_areas(boxes, keepdims=True):
    x_min, y_min, x_max, y_max = split(boxes, keepdims=keepdims)
    W = x_max - x_min
    H = y_max - y_min
    return W * H


def join(boxes):
    return jp.concatenate(boxes, axis=0)


def square(boxes):
    """Makes boxes square with sides equal to the longest original side.

    # Arguments
        box: Numpy array with shape `(4)` with point corner coordinates.

    # Returns
        returns: List of box coordinates ints.
    """
    x_min, y_min, x_max, y_max = split(boxes)
    center_x = (x_max + x_min) / 2.0
    center_y = (y_max + y_min) / 2.0
    boxes_W = x_max - x_min
    boxes_H = y_max - y_min
    x_min = jp.where(boxes_H >= boxes_W, center_x - (boxes_H / 2.0), x_min)
    x_max = jp.where(boxes_H >= boxes_W, center_x + (boxes_H / 2.0), x_max)
    y_min = jp.where(boxes_H >= boxes_W, y_min, center_y - (boxes_W / 2.0))
    y_max = jp.where(boxes_H >= boxes_W, y_max, center_y + (boxes_W / 2.0))
    return merge(x_min, y_min, x_max, y_max).astype(int)


def compute_sizes(boxes, keepdims=True):
    """Compute width and height from boxes in corner format."""
    x_min, y_min, x_max, y_max = split(boxes, keepdims)
    W = x_max - x_min
    H = y_max - y_min
    return H, W


def compute_centers(boxes):
    """Compute center coordinates of boxes."""
    x_min, y_min, x_max, y_max = split(boxes)
    center_x = (x_max + x_min) / 2.0
    center_y = (y_max + y_min) / 2.0
    return center_x, center_y


def to_center_form(boxes):
    """Convert bounding boxes from corner to center form.

    # Arguments:
        boxes (array): Boxes in corner format [x_min, y_min, x_max, y_max]

    # Returns:
        array: Boxes in center format [center_x, center_y, width, height]
    """

    center_x, center_y = compute_centers(boxes)
    H, W = compute_sizes(boxes)
    return jp.concatenate([center_x, center_y, W, H], axis=1)


def to_corner_form(boxes):
    """Convert bounding boxes from center to corner form.

    # Arguments:
        Boxes: Array of boxes in center format ``[center_x, center_y, W, H]``.

    # Returns:
        Boxes in corner format ``[x_min, y_min, x_max, y_max]``.
    """
    center_x, center_y, W, H = split(boxes)
    x_min = center_x - (W / 2.0)
    x_max = center_x + (W / 2.0)
    y_min = center_y - (H / 2.0)
    y_max = center_y + (H / 2.0)
    return jp.concatenate([x_min, y_min, x_max, y_max], axis=1)


def pad(boxes, size, value=-1):
    """Pad boxes array to specified size with value.

    # Arguments:
        boxes (Array): `(num_boxes, 4)` array of box coordinates.
        size (int): target size for padding.
        value (int): Value to pad with.

    # Returns:
        Padded boxes with shape `(size, 4)`.
    """
    num_boxes = len(boxes)
    if num_boxes > size:
        raise ValueError(f"Samples ({num_boxes}) exceeds pad ({size}).")
    padding = ((0, size - num_boxes), (0, 0))
    return jp.pad(boxes, padding, "constant", constant_values=value)


def from_selection(image, radius=5, color=(255, 0, 0), window_name="image"):
    """Manually select bounding boxes by double-clicking on an image."""
    points, boxes = [], []

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


def compute_IOU(box_A, boxes_B):
    """Computes intersection over union between `box_A` and `boxes_B`.

    # Arguments
        box_A: JAX Numpy array with shape `(4,)` representing a single box
            in corner form (x_min, y_min, x_max, y_max).
        boxes_B: JAX Numpy array with shape `(num_boxes_b, 4)` representing
            multiple boxes in corner form.

    # Returns
        JAX Numpy array of shape `(num_boxes_b,)` containing IoUs.
    """
    xy_min_inter = jp.maximum(box_A[0:2], boxes_B[:, 0:2])
    xy_max_inter = jp.minimum(box_A[2:4], boxes_B[:, 2:4])
    inter_wh = jp.maximum(0.0, xy_max_inter - xy_min_inter)
    intersection_area = inter_wh[:, 0] * inter_wh[:, 1]
    area_a = (box_A[2] - box_A[0]) * (box_A[3] - box_A[1])
    areas_b = (boxes_B[:, 2] - boxes_B[:, 0]) * (boxes_B[:, 3] - boxes_B[:, 1])
    union_area = (area_a + areas_b) - intersection_area
    union_area = jp.maximum(union_area, 1e-8)
    iou = intersection_area / union_area
    return jp.clip(iou, 0.0, 1.0)


def compute_IOUs(boxes_A, boxes_B):
    """Computes intersection over union (IOU) between `boxes_A` and `boxes_B`.

    For each box (rows `boxes_A`) it computes the IOU to all `boxes_B`.

    # Arguments
        boxes_A: Numpy array with shape `(num_boxes_A, 4)` in corner form.
        boxes_B: Numpy array with shape `(num_boxes_B, 4)` in corner form.

    # Returns
        Numpy array of shape `(num_boxes_A, num_boxes_B)`.
    """
    xy_min = jp.maximum(boxes_A[:, None, 0:2], boxes_B[:, 0:2])
    xy_max = jp.minimum(boxes_A[:, None, 2:4], boxes_B[:, 2:4])
    intersection = jp.maximum(0.0, xy_max - xy_min)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    areas_A = (boxes_A[:, 2] - boxes_A[:, 0]) * (boxes_A[:, 3] - boxes_A[:, 1])
    areas_B = (boxes_B[:, 2] - boxes_B[:, 0]) * (boxes_B[:, 3] - boxes_B[:, 1])
    # broadcasting for outer sum i.e. a sum of all possible combinations
    union_area = (areas_A[:, jp.newaxis] + areas_B) - intersection_area
    union_area = jp.maximum(union_area, 1e-8)
    return jp.clip(intersection_area / union_area, 0.0, 1.0)


def apply_NMS(boxes, scores, iou_thresh=0.45, top_k=200):
    top_k = min(top_k, len(boxes))
    sorted_score_args = jp.argsort(scores)[::-1]
    top_k_score_args = sorted_score_args[:top_k]
    top_k_boxes = jp.take(boxes, top_k_score_args, axis=0)
    top_k_args = jp.arange(len(top_k_boxes))

    def step(suppressed_mask, top_k_arg):
        is_suppressed = suppressed_mask[top_k_arg]

        def suppress():
            current_box = top_k_boxes[top_k_arg]
            current_box = jp.expand_dims(current_box, 0)
            ious = compute_IOUs(current_box, top_k_boxes)
            ious = jp.squeeze(ious, 0)
            mask_to_suppress = (ious > iou_thresh) & (top_k_args != top_k_arg)
            return jp.logical_or(suppressed_mask, mask_to_suppress)

        def do_nothing():
            return suppressed_mask

        new_suppressed_mask = jax.lax.cond(is_suppressed, do_nothing, suppress)
        keep_this_box = jp.logical_not(is_suppressed)
        return new_suppressed_mask, keep_this_box

    initial_mask = jp.zeros(len(top_k_boxes), dtype=bool)
    _, keep_mask = jax.lax.scan(step, initial_mask, top_k_args)
    selected_indices = top_k_score_args[keep_mask]
    return selected_indices


def encode(matched, priors, variances=[0.1, 0.1, 0.2, 0.2]):
    """Encode matched bounding boxes relative to prior boxes.

    # Arguments
        matched (array): Matched target boxes (assumed [cx, cy, w, h, extras...])
                         or converted by to_center_form if input is corner form.
                         Original code implies input `matched` needs conversion.
        priors (array): Prior boxes (assumed [cx, cy, w, h])
        variances (list): Variances for encoding (cx, cy, w, h)

    # Returns
        array: Encoded box parameters ([dx, dy, dw, dh, extras...])
    """

    def encode_centers(boxes_center, priors, variances):
        """Encode center coordinates using priors and variances."""
        difference_x = boxes_center[:, 0:1] - priors[:, 0:1]
        difference_y = boxes_center[:, 1:2] - priors[:, 1:2]
        encoded_center_x = (difference_x / priors[:, 2:3]) / variances[0]
        encoded_center_y = (difference_y / priors[:, 3:4]) / variances[1]
        return encoded_center_x, encoded_center_y

    def encode_sizes(boxes_center, priors, variances):
        """Encode width and height dimensions."""
        ratio_w = boxes_center[:, 2:3] / priors[:, 2:3]
        ratio_h = boxes_center[:, 3:4] / priors[:, 3:4]
        encoded_w = jp.log(ratio_w + 1e-8) / variances[2]
        encoded_h = jp.log(ratio_h + 1e-8) / variances[3]
        return encoded_w, encoded_h

    boxes_corner = matched[:, :4]
    boxes_center = to_center_form(boxes_corner)
    extras = matched[:, 4:]

    priors_center = priors

    encoded_x, encoded_y = encode_centers(
        boxes_center, priors_center, variances
    )
    encoded_w, encoded_h = encode_sizes(boxes_center, priors_center, variances)

    return jp.concatenate(
        [encoded_x, encoded_y, encoded_w, encoded_h, extras], axis=1
    )


def decode(predictions, priors, variances=[0.1, 0.1, 0.2, 0.2]):
    """Decode predicted box parameters to actual coordinates.

    # Arguments
        predictions (array): Encoded box predictions ([dx, dy, dw, dh, extras...])
        priors (array): Prior boxes (assumed [cx, cy, w, h])
        variances (list): Decoding variances

    # Returns
        array: Decoded boxes in corner format with extras ([xmin, ymin, xmax, ymax, extras...])
    """

    def decode_center_form_boxes(predictions, priors, variances):
        """Compute center-form boxes from predictions."""

        def decode_center_x(predictions, priors, variances):
            """Decode center x-coordinate from predictions."""
            return (
                predictions[:, 0:1] * priors[:, 2:3] * variances[0]
                + priors[:, 0:1]
            )

        def decode_center_y(predictions, priors, variances):
            """Decode center y-coordinate from predictions."""
            return (
                predictions[:, 1:2] * priors[:, 3:4] * variances[1]
                + priors[:, 1:2]
            )

        def decode_W(predictions, priors, variances):
            """Decode width from predictions."""
            exp_term = predictions[:, 2:3] * variances[2]
            return priors[:, 2:3] * jp.exp(exp_term)

        def decode_H(predictions, priors, variances):
            """Decode height from predictions."""
            exp_term = predictions[:, 3:4] * variances[3]
            return priors[:, 3:4] * jp.exp(exp_term)

        center_x = decode_center_x(predictions, priors, variances)
        center_y = decode_center_y(predictions, priors, variances)
        W = decode_W(predictions, priors, variances)
        H = decode_H(predictions, priors, variances)

        return jp.concatenate([center_x, center_y, W, H], axis=1)

    priors_center = priors
    boxes_center = decode_center_form_boxes(
        predictions, priors_center, variances
    )
    boxes_corner = to_corner_form(boxes_center)

    return jp.concatenate([boxes_corner, predictions[:, 4:]], axis=1)
