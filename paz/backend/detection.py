import jax
import jax.numpy as jp
import paz


def random_flip_left_right(image, boxes):
    boxes = paz.boxes.flip_left_right(boxes, image.shape[1])
    image = paz.image.flip_left_right(image)
    return image, boxes


def split(detections):
    boxes = detections[:, :4]
    class_args = detections[:, 4:]
    return boxes, class_args


def get_boxes(detections):
    return detections[:, :4]


def get_scores(detections):
    return detections[:, 4:]


def merge(boxes, class_args):
    return jp.concatenate([boxes, class_args], axis=1)


def to_one_hot(detections, num_classes):
    boxes, classes = split(detections)
    classes = paz.classes.to_one_hot(classes, num_classes)
    return merge(boxes, classes)


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


def select_top_k(boxes_and_scores, top_k=200):
    boxes, scores = paz.detection.split(boxes_and_scores)
    sorted_score_args = jp.argsort(jp.squeeze(scores, axis=-1))[::-1]
    top_k_score_args = sorted_score_args[:top_k]
    return boxes_and_scores[top_k_score_args]


def to_score(boxes_and_one_hot_vectors, class_arg):
    boxes, one_hot_vectors = paz.detection.split(boxes_and_one_hot_vectors)
    class_scores = jp.expand_dims(one_hot_vectors[:, class_arg], 1)
    boxes_and_scores = paz.detection.merge(boxes, class_scores)
    return boxes_and_scores


def to_one_hot_vector(boxes_and_scores, class_arg, num_classes):
    boxes, scores = paz.detection.split(boxes_and_scores)
    one_hot_vectors = jp.zeros((len(boxes), num_classes))
    scores = jp.squeeze(scores, axis=-1)
    one_hot_vectors = one_hot_vectors.at[:, class_arg].set(scores)
    boxes_and_one_hot_vectors = paz.detection.merge(boxes, one_hot_vectors)
    return boxes_and_one_hot_vectors


def apply_NMS(sorted_boxes_with_scores, iou_thresh=0.45, epsilon=0.01):
    top_k_boxes = paz.detection.get_boxes(sorted_boxes_with_scores)
    top_k_boxes_args = jp.arange(len(top_k_boxes))
    num_total_boxes = top_k_boxes.shape[0]

    def do_continue(state):
        suppressed_mask, top_k_box_arg = state
        in_bounds = top_k_box_arg < num_total_boxes

        def any_unprocessed_unsuppressed():
            is_in_suffix_to_check = top_k_boxes_args >= top_k_box_arg
            is_unsuppressed = jp.logical_not(suppressed_mask)
            unsuppressed_in_suffix = jp.logical_and(
                is_unsuppressed, is_in_suffix_to_check
            )
            return jp.any(unsuppressed_in_suffix)

        return jax.lax.cond(
            in_bounds, any_unprocessed_unsuppressed, lambda: False
        )

    def step(state):
        suppressed_mask, top_k_box_arg = state
        is_suppressed = suppressed_mask[top_k_box_arg]

        def suppress():
            current_box = top_k_boxes[top_k_box_arg]
            ious = paz.boxes.compute_IOU(current_box, top_k_boxes)
            is_not_this_box = top_k_boxes_args != top_k_box_arg
            do_suppress = (ious > iou_thresh) & is_not_this_box
            return jp.logical_or(suppressed_mask, do_suppress)

        def do_nothing():
            return suppressed_mask

        new_suppressed_mask = jax.lax.cond(is_suppressed, do_nothing, suppress)
        return (new_suppressed_mask, top_k_box_arg + 1)

    scores = jp.squeeze(paz.detection.get_scores(sorted_boxes_with_scores), -1)
    suppressed_mask = scores < epsilon
    state = (suppressed_mask, 0)
    suppressed_mask, num_steps = jax.lax.while_loop(do_continue, step, state)
    return jp.logical_not(suppressed_mask)  # keep mask


def single_class(
    class_arg, detections, num_classes, iou_thresh, top_k, epsilon
):
    class_detections = to_score(detections, class_arg)
    class_detections = select_top_k(class_detections, top_k)
    non_suppressed = apply_NMS(class_detections, iou_thresh)
    valid_scores = split(class_detections)[1] >= epsilon
    non_suppressed = jp.expand_dims(non_suppressed, axis=-1)
    class_keep_mask = jp.logical_and(valid_scores, non_suppressed)
    class_detections = to_one_hot_vector(
        class_detections, class_arg, num_classes
    )
    return class_detections, class_keep_mask


def apply_per_class_NMS(
    detections,
    num_classes,
    iou_thresh=0.45,
    top_k=200,
    epsilon=0.01,
):
    multi_class = jax.vmap(
        single_class,
        in_axes=(0, None, None, None, None, None),
    )

    detections, keep_masks = multi_class(
        jp.arange(num_classes),
        detections,
        num_classes,
        iou_thresh,
        top_k,
        epsilon,
    )
    # detections = detections.reshape(-1, detections.shape[-1])
    detections = detections.reshape(-1, 6)
    keep_masks = keep_masks.reshape(-1, 1)
    detections = jp.where(keep_masks, detections, -1.0)
    return detections


def filter_by_score(detections, threshold):
    """Filters detections by scores."""
    scores = jp.max(paz.detection.get_scores(detections), axis=1, keepdims=True)
    return jp.where(scores >= threshold, detections, -1)


def remove_class(detections, class_arg):
    """Remove a particular class from the pipeline.

    # Arguments
        class_names: List, indicating given class names.
        class_arg: Int, index of the class to be removed.
    """
    # del class_names[class_arg]
    return jp.delete(detections, 4 + class_arg, axis=1)


def denormalize(detections, image_shape):
    boxes, scores = split(detections)
    boxes = paz.boxes.denormalize(boxes, image_shape)
    return merge(boxes, scores)


def remove_invalid(detections_array, value=-1):
    is_invalid_row_mask = jp.all(detections_array == value, axis=1)
    is_valid_row_mask = jp.logical_not(is_invalid_row_mask)
    valid_boxes = detections_array[is_valid_row_mask]
    return valid_boxes


def to_boxes2D(boxes_and_one_hot_vectors):
    boxes, one_hot_vectors = split(boxes_and_one_hot_vectors)
    class_args = jp.argmax(one_hot_vectors, axis=1)
    scores = one_hot_vectors[class_args]
    return jp.concatenate([boxes, class_args, scores], axis=1)
