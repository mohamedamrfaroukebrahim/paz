import jax
import jax.numpy as jp
import cv2
import paz


def split(boxes, keepdims=True, axis=1):
    """Split boxes into x_min, y_min, x_max, y_max components."""
    coordinates = jp.split(boxes, 4, axis=axis)
    if keepdims:
        return coordinates
    else:
        return tuple(jp.squeeze(column, axis=-1) for column in coordinates)


def compute_centers(boxes):
    """Compute center coordinates of boxes."""
    x_min, y_min, x_max, y_max = split(boxes)
    center_x = (x_max + x_min) / 2.0
    center_y = (y_max + y_min) / 2.0
    return center_x, center_y


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


def compute_sizes(boxes):
    """Compute width and height from boxes in corner format."""
    x_min, y_min, x_max, y_max = split(boxes)
    W = x_max - x_min
    H = y_max - y_min
    return H, W


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


def to_xywh(boxes):
    return xyxy_to_xywh(boxes)


def xyxy_to_xywh(boxes):
    x_min, y_min = boxes[:, 0:1], boxes[:, 1:2]
    x_max, y_max = boxes[:, 2:3], boxes[:, 3:4]
    W = x_max - x_min
    H = y_max - y_min
    return merge(x_min, y_min, W, H)


def xywh_to_xyxy(boxes):
    x_min, y_min, W, H = split(boxes)
    x_max = x_min + W
    y_max = y_min + H
    boxes = merge(x_min, y_min, x_max, y_max)
    return boxes


def flip_left_right(boxes, image_width):
    """Flips box coordinates from left-to-right and vice-versa.

    # Arguments
        boxes: Numpy array of shape `(num_boxes, 4)`.

    # Returns
        Numpy array of shape `(num_boxes, 4)`.
    """
    x_min, y_min, x_max, y_max = split(boxes)
    return merge(x_max, y_min, x_min, y_max)


def append_class(boxes, class_arg):
    class_args = jp.full((len(boxes), 1), class_arg)
    return jp.hstack((boxes, class_args))
