import numpy as np
import jax.numpy as jp
import paz


def match(boxes, prior_boxes, IOU_threshold=0.5):
    """Matches each prior box with a ground truth box (box from `boxes`).
    It then selects which matched box will be considered positive e.g. iou > .5
    and returns for each prior box a ground truth box that is either positive
    (with a class argument different than 0) or negative.

    # Arguments
        boxes: Numpy array of shape `(num_ground_truh_boxes, 4 + 1)`,
            where the first the first four coordinates correspond to
            box coordinates and the last coordinates is the class
            argument. This boxes should be the ground truth boxes.
        prior_boxes: Numpy array of shape `(num_prior_boxes, 4)`.
            where the four coordinates are in center form coordinates.
        iou_threshold: Float between [0, 1]. Intersection over union
            used to determine which box is considered a positive box.

    # Returns
        Array of shape `(num_prior_boxes, 4 + 1)`.
            where the first the first four coordinates correspond to point
            form box coordinates and the last coordinates is the class
            argument.
    """

    def mark_best_match(per_prior_best_IOU, per_box_best_prior_arg):
        # The prior boxes that are the best match for each box are marked.
        # They are marked by setting an IOU larger (2) than the maxium (1).
        # the best prior box match of box_0 is per_box_best_prior_arg[0]
        # the best prior box match of box_1 is per_box_best_prior_arg[1]
        # ...
        return per_prior_best_IOU.at[per_box_best_prior_arg].set(2.0)

    def select_for_each_prior_box_a_box(boxes, per_prior_best_box):
        # Each prior box is assigned a ground truth box.
        assigned_boxes = boxes[per_prior_best_box]
        return assigned_boxes

    def force_match(per_prior_best_box, per_box_best_prior):
        # Ensures that every ground truth box is matched with at least one prior
        # box. Specifically, the prior box with which it has the highest IoU.
        for box_arg, prior_arg in enumerate(per_box_best_prior):
            per_prior_best_box = per_prior_best_box.at[prior_arg].set(box_arg)
        return per_prior_best_box

    def label_negative_boxes(assigned_boxes, per_prior_best_IOU):
        is_low_IOU_match = per_prior_best_IOU < IOU_threshold
        class_args = assigned_boxes[:, 4]
        class_args = jp.where(is_low_IOU_match, 0.0, class_args)
        return assigned_boxes.at[:, 4].set(class_args)

    prior_boxes = paz.boxes.to_corner_form(prior_boxes)
    IOUs = paz.boxes.compute_IOUs(boxes, prior_boxes)  # (boxes, prior_boxes)
    per_box_best_prior = jp.argmax(IOUs, axis=1)  # (boxes,)
    per_prior_best_box = jp.argmax(IOUs, axis=0)  # (prior_boxes,)
    per_prior_best_IOU = jp.max(IOUs, axis=0)  # (prior_boxes,)
    per_prior_best_IOU = mark_best_match(per_prior_best_IOU, per_box_best_prior)
    assign_args = (per_prior_best_box, per_box_best_prior)
    per_prior_best_box = force_match(*assign_args)
    selected_boxes = select_for_each_prior_box_a_box(boxes, per_prior_best_box)
    selected_boxes = label_negative_boxes(selected_boxes, per_prior_best_IOU)
    return selected_boxes


if __name__ == "__main__":
    from prior_boxes import create_prior_boxes

    prior_boxes = create_prior_boxes("VOC")
    box_with_label = jp.array(
        [
            [47.0, 239.0, 194.0, 370.0, 12.0],
            [7.0, 11.0, 351.0, 497.0, 15.0],
            [138.0, 199.0, 206.0, 300.0, 19.0],
            [122.0, 154.0, 214.0, 194.0, 18.0],
            [238.0, 155.0, 306.0, 204.0, 9.0],
        ]
    )

    matches_old = match_old(box_with_label, prior_boxes)
    matches_now = match(box_with_label, prior_boxes)
