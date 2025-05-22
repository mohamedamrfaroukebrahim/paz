import jax
import jax.numpy as jp
import paz


def select_top_k(boxes_and_scores, top_k=200):
    boxes, scores = paz.detection.split(boxes_and_scores)
    top_k = jp.minimum(top_k, len(boxes))
    sorted_score_args = jp.argsort(scores)[::-1]
    top_k_score_args = sorted_score_args[:top_k]
    return jp.take(boxes_and_scores, top_k_score_args, axis=0)


def to_score(boxes_and_one_hot_vectors, class_arg):
    boxes, one_hot_vectors = paz.detection.split(boxes_and_one_hot_vectors)
    class_scores = one_hot_vectors[:, class_arg]
    boxes_and_scores = paz.detection.join(boxes, class_scores)
    return boxes_and_scores


def to_one_hot_vector(boxes_and_scores, class_arg, num_classes):
    boxes, scores = paz.detection.split(boxes_and_scores)
    one_hot_vectors = jp.zeros((len(boxes), num_classes))
    one_hot_vectors = one_hot_vectors.at[:, class_arg].set(scores)
    boxes_and_one_hot_vectors = paz.detection.join(boxes, one_hot_vectors)
    return boxes_and_one_hot_vectors


def apply_NMS(sorted_boxes_with_scores, iou_thresh=0.45):
    top_k_boxes = paz.detection.get_boxes(sorted_boxes_with_scores)
    top_k_boxes_args = jp.arange(len(top_k_boxes))

    def step(suppressed_mask, top_k_arg):
        is_suppressed = suppressed_mask[top_k_arg]

        def suppress():
            current_box = top_k_boxes[top_k_arg]
            current_box = jp.expand_dims(current_box, 0)
            ious = paz.boxes.compute_IOUs(current_box, top_k_boxes)
            ious = jp.squeeze(ious, 0)
            is_not_this_box = top_k_boxes_args != top_k_arg
            do_suppress = (ious > iou_thresh) & is_not_this_box
            return jp.logical_or(suppressed_mask, do_suppress)

        def do_nothing():
            return suppressed_mask

        new_suppressed_mask = jax.lax.cond(is_suppressed, do_nothing, suppress)
        keep_this_box = jp.logical_not(is_suppressed)
        return new_suppressed_mask, keep_this_box

    initial_mask = jp.zeros(len(top_k_boxes), dtype=bool)
    _, keep_mask = jax.lax.scan(step, initial_mask, top_k_boxes_args)
    return keep_mask


def apply_per_class_NMS(
    detections, num_classes, iou_thresh=0.45, epsilon=0.01, top_k=200
):
    # TODO fix destruction of class distribution by "to_score" and "to_vector".
    non_suppressed_detections, keep_masks = [], []
    to_vector = paz.lock(to_one_hot_vector, num_classes)
    for class_arg in range(num_classes):
        class_detections = to_score(detections, class_arg)
        class_detections = select_top_k(class_detections, top_k)
        non_suppressed = apply_NMS(class_detections, iou_thresh)
        valid_scores = paz.detection.split(class_detections)[1] >= epsilon
        class_keep_mask = jp.logical_and(valid_scores, non_suppressed)
        class_detections = to_vector(class_detections, class_arg)
        non_suppressed_detections.append(class_detections)
        keep_masks.append(class_keep_mask)
    keep_masks = jp.concatenate(keep_masks, 0)
    non_suppressed_detections = jp.concatenate(non_suppressed_detections, 0)
    return jp.where(keep_masks, non_suppressed_detections, -1)


def filter_by_score(detections, threshold):
    """Filters detections by scores."""
    scores = jp.max(paz.boxes.get_scores(detections), axis=1)
    return jp.where(scores >= threshold, detections, -1)
