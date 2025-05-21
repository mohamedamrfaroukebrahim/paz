import numpy as np
import jax.numpy as jp
import paz


def random_flip_left_right(image, boxes):
    boxes = paz.boxes.flip_left_right(boxes, image.shape[1])
    image = paz.image.flip_left_right(image)
    return image, boxes


def non_max_suppression(boxes, scores, iou_thresh=0.45, top_k=200):
    selected_indices = np.zeros(shape=len(scores))
    if boxes is None or len(boxes) == 0:
        return selected_indices
    x_min = boxes[:, 0]
    y_min = boxes[:, 1]
    x_max = boxes[:, 2]
    y_max = boxes[:, 3]
    areas = (x_max - x_min) * (y_max - y_min)
    remaining_sorted_box_indices = np.argsort(scores)
    remaining_sorted_box_indices = remaining_sorted_box_indices[-top_k:]

    num_selected_boxes = 0
    while len(remaining_sorted_box_indices) > 0:
        best_score_args = remaining_sorted_box_indices[-1]
        selected_indices[num_selected_boxes] = best_score_args
        num_selected_boxes = num_selected_boxes + 1
        if len(remaining_sorted_box_indices) == 1:
            break

        remaining_sorted_box_indices = remaining_sorted_box_indices[:-1]

        best_x_min = x_min[best_score_args]
        best_y_min = y_min[best_score_args]
        best_x_max = x_max[best_score_args]
        best_y_max = y_max[best_score_args]

        remaining_x_min = x_min[remaining_sorted_box_indices]
        remaining_y_min = y_min[remaining_sorted_box_indices]
        remaining_x_max = x_max[remaining_sorted_box_indices]
        remaining_y_max = y_max[remaining_sorted_box_indices]

        inner_x_min = np.maximum(remaining_x_min, best_x_min)
        inner_y_min = np.maximum(remaining_y_min, best_y_min)
        inner_x_max = np.minimum(remaining_x_max, best_x_max)
        inner_y_max = np.minimum(remaining_y_max, best_y_max)

        inner_box_widths = inner_x_max - inner_x_min
        inner_box_heights = inner_y_max - inner_y_min

        inner_box_widths = np.maximum(inner_box_widths, 0.0)
        inner_box_heights = np.maximum(inner_box_heights, 0.0)

        intersections = inner_box_widths * inner_box_heights
        remaining_box_areas = areas[remaining_sorted_box_indices]
        best_area = areas[best_score_args]
        unions = remaining_box_areas + best_area - intersections
        intersec_over_union = intersections / unions
        intersec_over_union_mask = intersec_over_union <= iou_thresh
        remaining_sorted_box_indices = remaining_sorted_box_indices[
            intersec_over_union_mask
        ]

    return selected_indices.astype(int), num_selected_boxes


def split(detections):
    # TODO assumes class args are the last column
    boxes = detections[:, :4]
    class_args = detections[:, 4:5]
    return boxes, class_args


def get_boxes(detections):
    return detections[:, :4]


def join(boxes, class_args):
    return jp.concatenate([boxes, class_args], axis=0)


def to_one_hot(detections, num_classes):
    boxes, classes = split(detections)
    classes = paz.classes.to_one_hot(classes, num_classes)
    return join(boxes, classes)


def encode(detections, priors, variances=[0.1, 0.1, 0.2, 0.2]):
    boxes = detections[:, :4]
    boxes = paz.boxes.to_center_form(boxes)
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
    return jp.concatenate(encoded_boxes + [detections[:, 4:]], axis=1)


def decode(predictions, priors, variances=[0.1, 0.1, 0.2, 0.2]):
    center_x = predictions[:, 0:1] * priors[:, 2:3] * variances[0]
    center_x = center_x + priors[:, 0:1]
    center_y = predictions[:, 1:2] * priors[:, 3:4] * variances[1]
    center_y = center_y + priors[:, 1:2]
    W = priors[:, 2:3] * jp.exp(predictions[:, 2:3] * variances[2])
    H = priors[:, 3:4] * jp.exp(predictions[:, 3:4] * variances[3])
    boxes = jp.concatenate([center_x, center_y, W, H], axis=1)
    boxes = paz.boxes.to_corner_form(boxes)
    return jp.concatenate([boxes, predictions[:, 4:]], 1)
