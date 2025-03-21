import jax
from jax import lax
import jax.numpy as jp
from paz.backend.boxes import compute_iou
import types


def should_apply_crop(probability, key):
    """
    Determines whether to apply a random crop based on the given probability.
    """
    r = jax.random.uniform(key, shape=())
    return probability >= r


def adjust_boxes(boxes, labels, image_crop_box, mask):
    """
    Adjusts bounding boxes and their associated labels to fit within a cropped region.
    """
    masked_boxes = boxes[mask]
    masked_labels = labels[mask]
    masked_boxes = masked_boxes.at[:, :2].set(
        jp.maximum(masked_boxes[:, :2], image_crop_box[:2]) - image_crop_box[:2]
    )
    masked_boxes = masked_boxes.at[:, 2:].set(
        jp.minimum(masked_boxes[:, 2:], image_crop_box[2:]) - image_crop_box[:2]
    )
    return jp.hstack([masked_boxes, masked_labels])


def get_random_crop(W_original, H_original, key):
    """
    Generates a random crop region from the original image dimensions.
    """
    min_W = jp.maximum(1, jp.floor(0.3 * W_original)).astype(jp.int32)
    max_W = W_original
    min_H = jp.maximum(1, jp.floor(0.3 * H_original)).astype(jp.int32)
    max_H = H_original

    key, subkey = jax.random.split(key)
    W = jax.random.randint(subkey, (), min_W, max_W, dtype=jp.int32)

    key, subkey = jax.random.split(key)
    H = jax.random.randint(subkey, (), min_H, max_H, dtype=jp.int32)

    aspect_ratio = H / W
    valid = jp.logical_and(aspect_ratio >= 0.5, aspect_ratio <= 2.0)

    key, subkey = jax.random.split(key)
    x_min = jax.random.randint(subkey, (), 0, W_original - W, dtype=jp.int32)

    key, subkey = jax.random.split(key)
    y_min = jax.random.randint(subkey, (), 0, H_original - H, dtype=jp.int32)

    crop_box = jp.where(
        valid,
        jp.array([x_min, y_min, x_min + W, y_min + H]),
        jp.array([0, 0, 0, 0]),
    )

    return crop_box, key


def compute_valid_mask(boxes, image_crop_box, min_iou, max_iou):
    """
    Compute the IoU overlap between the crop and bounding boxes,
    and create a mask for valid boxes using JAX.
    """
    overlap = compute_iou(image_crop_box, boxes)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    mask = (
        (image_crop_box[0] < centers[:, 0])
        & (image_crop_box[1] < centers[:, 1])
        & (image_crop_box[2] > centers[:, 0])
        & (image_crop_box[3] > centers[:, 1])
    )
    iou_condition = (jp.max(overlap) >= min_iou) & (jp.min(overlap) <= max_iou)
    mask_condition = jp.any(mask)
    is_valid_mask = lax.cond(
        iou_condition & mask_condition, lambda _: True, lambda _: False, operand=None
    )
    return is_valid_mask, mask


def crop_and_adjust_boxes(
    image, labels, key, max_trials, boxes, crop_box, min_iou, max_iou, final_boxes
):
    """
    Crop the image using the provided crop_box and adjust bounding
    boxes if the IoU constraints are met.
    """
    is_valid_mask, mask = compute_valid_mask(boxes, crop_box, min_iou, max_iou)
    trials = 0

    def loop_condition_fn(state):
        trials, is_valid_mask, _ = state
        return (trials < max_trials) & (~is_valid_mask)

    def loop_body_fn(state):
        trials, _, _ = state
        new_is_valid_mask, new_mask = compute_valid_mask(boxes, crop_box, min_iou, max_iou)
        return (trials + 1, new_is_valid_mask, new_mask)

    final_state = lax.while_loop(loop_condition_fn, loop_body_fn, (trials, is_valid_mask, mask))
    final_trials, final_is_valid_mask, final_mask = final_state

    if final_trials != max_trials and final_is_valid_mask:
        cropped_image = image[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2], :]
        cropped_boxes = adjust_boxes(boxes, labels, crop_box, final_mask)
        image, final_boxes, key = cropped_image, cropped_boxes, key

    return image, final_boxes, key


def find_valid_crop(
    image,
    labels,
    max_trials,
    W_original,
    H_original,
    key,
    boxes,
    min_iou,
    max_iou,
    final_boxes,
):
    """
    Attempts to generate a valid cropped region from the original image while
    ensuring bounding box validity.
    """
    final_image = image
    crop_box, key = get_random_crop(W_original, H_original, key)
    if crop_box is not None:
        final_image, final_boxes, key = crop_and_adjust_boxes(
            image, labels, key, max_trials, boxes, crop_box, min_iou, max_iou, final_boxes
        )
    return final_image, final_boxes, key


def get_boxes_labels_mode(image, boxes, jaccard_min_max, key):
    """
    Extract bounding boxes and labels from the image and randomly select an IoU mode.
    """
    key, subkey = jax.random.split(key)

    labels = boxes[:, -1:]
    bounding_boxes = boxes[:, :4]

    mode = jax.random.randint(subkey, shape=(), minval=0, maxval=len(jaccard_min_max))

    return labels, bounding_boxes, mode


def attempt_random_crop(image, labels, bounding_boxes, max_trials, iou_range, key, final_boxes):
    """
    Find a valid random crop of the image that satisfies the given IoU range.
    """
    H_original, W_original = image.shape[:2]
    min_iou, max_iou = iou_range
    return find_valid_crop(
        image,
        labels,
        max_trials,
        W_original,
        H_original,
        key,
        bounding_boxes,
        min_iou,
        max_iou,
        final_boxes,
    )


def random_sample_crop_fn(image, boxes, probability, max_trials, jaccard_min_max, key):
    """
    Randomly crop the image (based on a specified probability) and adjust the bounding boxes.
    """
    final_boxes = boxes
    key, subkey = jax.random.split(key)

    if not should_apply_crop(probability, subkey):
        return image, boxes, key

    key, subkey = jax.random.split(key)

    labels, bounding_boxes, mode = get_boxes_labels_mode(image, boxes, jaccard_min_max, subkey)

    if jaccard_min_max[mode] is None:
        final_boxes = jp.hstack([bounding_boxes, labels])
    else:
        return attempt_random_crop(
            image, labels, bounding_boxes, max_trials, jaccard_min_max[mode], key, final_boxes
        )

    return image, boxes, key


def RandomSampleCrop(probability=0.50, max_trials=50, seed=0):
    """
    Creates a callable processor for performing random sample cropping on images with bounding
    boxes.

    Args:
        probability (float): The probability of applying the random crop operation (default: 0.5).
        max_trials (int): Maximum number of attempts to find a valid crop region (default: 50).
        seed (int): Seed for the random number generator to ensure reproducibility (default: 0).

    Returns:
        A callable object (`processor`) with the following attributes:
            - call: A function that applies random cropping to an image and adjusts bounding boxes.
    """
    processor = types.SimpleNamespace(
        probability=probability,
        max_trials=max_trials,
        jaccard_min_max=(
            None,
            (0.1, jp.inf),
            (0.3, jp.inf),
            (0.7, jp.inf),
            (0.9, jp.inf),
            (-jp.inf, jp.inf),
        ),
        key=jax.random.PRNGKey(seed),
    )

    def call(image, boxes):
        """
        Applies random sample cropping to the input image and adjusts the bounding boxes.

        Args:
            image: The input image as a JAX array or NumPy array.
            boxes: Bounding boxes associated with the image. Each box is represented
                                as [x_min, y_min, x_max, y_max, label].

        Returns:
            tuple: A tuple containing:
                - new_image: The cropped image.
                - new_boxes : Adjusted bounding boxes corresponding to the
                                        cropped image.
        """

        probability = processor.probability
        max_trials = processor.max_trials
        jaccard_min_max = processor.jaccard_min_max
        key = processor.key

        new_image, new_boxes, new_key = random_sample_crop_fn(
            image, boxes, probability, max_trials, jaccard_min_max, key
        )

        processor.key = new_key
        return new_image, new_boxes

    processor.call = call
    return processor
