import cv2
import jax.numpy as jp
from jax import lax
import paz
import numpy as np


def split(boxes):
    return jp.split(boxes, 4, axis=1)


def merge(coordinate_0, coordinate_1, coordinate_2, coordinate_3):
    return jp.concatenate(
        [coordinate_0, coordinate_1, coordinate_2, coordinate_3], axis=1
    )


def compute_area(boxes):
    x_min, y_min, x_max, y_max = split(boxes)
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
    # TODO missing with edge cases
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
    return merge(x_max, y_min, x_min, y_max)


def from_selection(image, radius=5, color=(255, 0, 0), window_name="image"):
    points, boxes = [], []
    image = image.copy()
    image = np.array(image)

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


def calculate_IoU_with_best_box(x_min, x_max, y_min, y_max, best_idx, areas):
    """Calculates IoU of all boxes with the best box."""
    best_xmin = x_min[best_idx]
    best_ymin = y_min[best_idx]
    best_xmax = x_max[best_idx]
    best_ymax = y_max[best_idx]

    Intersection_xmin = jp.maximum(x_min, best_xmin)
    Intersection_ymin = jp.maximum(y_min, best_ymin)
    Intersection_xmax = jp.minimum(x_max, best_xmax)
    Intersection_ymax = jp.minimum(y_max, best_ymax)

    Intersection_width = jp.maximum(Intersection_xmax - Intersection_xmin, 0.0)
    Intersection_height = jp.maximum(Intersection_ymax - Intersection_ymin, 0.0)
    intersection = Intersection_width * Intersection_height

    union = areas + areas[best_idx] - intersection
    iou = intersection / union

    return iou


def _suppress_overlapping_boxes(best_idx, iou, curr_scores, iou_thresh=0.45):
    """Suppresses overlapping boxes based on IoU threshold."""
    suppressed_scores = jp.where(iou > iou_thresh, -jp.inf, curr_scores)
    updated_scores = suppressed_scores.at[best_idx].set(-jp.inf)
    return updated_scores


def _nms_iteration_step(x_min, y_min, x_max, y_max, areas, iou_thresh):
    """Performs one iteration of NMS, suppressing overlapping boxes."""

    def step(state):
        i, scores, indices = state
        best_idx = jp.argmax(scores).astype(jp.int32)
        updated_indices = indices.at[i].set(best_idx)
        iou = calculate_IoU_with_best_box(
            x_min, x_max, y_min, y_max, best_idx, areas
        )
        updated_scores = _suppress_overlapping_boxes(
            best_idx, iou, scores, iou_thresh
        )

        return (i + 1, updated_scores, updated_indices)

    return step


def _nms_continuation_condition(top_k):
    """Determines whether to continue the NMS iterations."""

    def condition(state):
        i, curr_scores, _ = state
        continue_iteration = jp.logical_and(
            i < top_k, jp.max(curr_scores) > -jp.inf
        )
        return continue_iteration

    return condition


def _compute_box_geometric_features(boxes):
    """
    Computes geometric features of boxes: x_min, y_min, x_max, y_max, areas.
    """
    x_min, y_min = boxes[:, 0], boxes[:, 1]
    x_max, y_max = boxes[:, 2], boxes[:, 3]
    areas = (x_max - x_min) * (y_max - y_min)
    return x_min, y_min, x_max, y_max, areas


def apply_non_max_suppression(boxes, scores, iou_thresh=0.45, top_k=200):
    """
    Applies Non-Maximum Suppression using fixed-iteration loop.

    Args:
        boxes: Numpy array of shape (num_boxes, 4) [x_min, y_min, x_max, y_max]
        scores: Numpy array of shape (num_boxes,) with confidence scores
        iou_thresh: IoU threshold for suppression
        top_k: Maximum number of boxes to select

    Returns:
        selected_indices: Numpy array of selected box indices
        num_selected: Number of selected boxes
    """
    num_boxes = boxes.shape[0]
    if num_boxes == 0:
        return jp.zeros(0, dtype=jp.int32), 0

    x_min, y_min, x_max, y_max, areas = _compute_box_geometric_features(boxes)

    selected_indices = jp.zeros(top_k, dtype=jp.int32)
    init_state = (0, scores, selected_indices)

    step_fn = _nms_iteration_step(x_min, y_min, x_max, y_max, areas, iou_thresh)
    condition_fn = _nms_continuation_condition(top_k)

    final_state = lax.while_loop(condition_fn, step_fn, init_state)
    num_selected, _, final_indices = final_state

    selected_indices = lax.dynamic_slice(
        final_indices, (0,), (jp.minimum(num_selected, top_k),)
    )
    return selected_indices, num_selected


def nms_per_class(box_data, nms_thresh=0.45, confidence_thresh=0.01, top_k=200):
    """
    Applies non-maximum-suppression per class.

    Args :
        box_data: Numpy array of shape `(num_prior_boxes, 4 + num_classes)`.
        nms_thresh: Float. Non-maximum suppression threshold.
        confidence_thresh: Float. Filter scores with a lower confidence value
            before performing non-maximum suppression.
        top_k: Integer. Maximum number of boxes per class outputted by NMS.

    Returns :
        Numpy array of shape `(num_classes, top_k, 5)`.
    """
    decoded_boxes, class_predictions = box_data[:, :4], box_data[:, 4:]
    num_classes = class_predictions.shape[1]
    output = jp.zeros((num_classes, top_k, 5))

    for class_arg in range(1, num_classes):
        confidence_mask = class_predictions[:, class_arg] >= confidence_thresh
        scores = class_predictions[:, class_arg][confidence_mask]
        if len(scores) == 0:
            continue
        boxes = decoded_boxes[confidence_mask]
        indices, count = apply_non_max_suppression(
            boxes, scores, nms_thresh, top_k
        )
        scores = jp.expand_dims(scores, -1)
        selected_indices = indices[:count]
        selections = jp.concatenate(
            (boxes[selected_indices], scores[selected_indices]), axis=1
        )
        output = output.at[class_arg, :count, :].set(selections)
    return output


def compute_iou(box, boxes):
    """Calculates the intersection over union between 'box' and all 'boxes'.
    Both `box` and `boxes` are in corner coordinates.

    # Arguments
        box: JAX array with length at least of 4.
        boxes: JAX array with shape `(num_boxes, 4)`.

    # Returns
        JAX array of shape `(num_boxes, 1)`.
    """

    x_min_A, y_min_A, x_max_A, y_max_A = box[:4]
    x_min_B, y_min_B = boxes[:, 0], boxes[:, 1]
    x_max_B, y_max_B = boxes[:, 2], boxes[:, 3]
    # calculating the intersection
    inner_x_min = jp.maximum(x_min_B, x_min_A)
    inner_y_min = jp.maximum(y_min_B, y_min_A)
    inner_x_max = jp.minimum(x_max_B, x_max_A)
    inner_y_max = jp.minimum(y_max_B, y_max_A)
    inner_w = jp.maximum((inner_x_max - inner_x_min), 0)
    inner_h = jp.maximum((inner_y_max - inner_y_min), 0)
    intersection_area = inner_w * inner_h
    # calculating the union
    box_area_B = (x_max_B - x_min_B) * (y_max_B - y_min_B)
    box_area_A = (x_max_A - x_min_A) * (y_max_A - y_min_A)
    union_area = box_area_A + box_area_B - intersection_area
    intersection_over_union = intersection_area / union_area
    return intersection_over_union


def append_class(boxes, class_arg):
    class_args = jp.full((len(boxes), 1), class_arg)
    return jp.hstack((boxes, class_args))
