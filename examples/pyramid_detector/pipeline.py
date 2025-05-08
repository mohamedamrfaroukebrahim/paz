import jax
import jax.numpy as jp
import paz


def compute_samples(positive_ratio, batch_size):
    num_positives = int(positive_ratio * batch_size)
    num_negatives = batch_size - num_positives
    return num_positives, num_negatives


def augment(key, image, angle_range=(-jp.pi / 8, jp.pi / 8)):
    key_0, key_1, key_2, key_3, key_4, key_5 = jax.random.split(key, 6)
    image = paz.image.random_flip_left_right(key_0, image)
    image = paz.image.random_saturation(key_1, image)
    image = paz.image.random_brightness(key_2, image)
    image = paz.image.random_contrast(key_3, image, 0.9, 1.1)
    image = paz.image.random_hue(key_4, image)
    image = paz.image.random_rotation(key_5, image, *angle_range)
    return image


def build_labels(key, positive_images, negative_images):
    positive_labels = jp.full(len(positive_images), 1.0)
    negative_labels = jp.full(len(negative_images), 0.0)
    images = jp.concatenate([negative_images, positive_images], axis=0)
    labels = jp.concatenate([negative_labels, positive_labels], axis=0)
    shuffled_args = jax.random.permutation(key, jp.arange(len(images)))
    images = images[shuffled_args]
    labels = labels[shuffled_args]
    return images, labels


def batch(
    key,
    detections,
    image,
    box_size=(128, 128),
    positive_ratio=0.5,
    batch_size=32,
    scale_range=(0.8, 1.4),
    shift_range=(-20, 20),
    pad=jp.array(paz.image.RGB_IMAGENET_MEAN, dtype="uint8"),
):
    keys = jax.random.split(key, 5)
    size = paz.image.get_size(image)
    positive_boxes = paz.detection.get_boxes(detections)
    positive_boxes = paz.boxes.denormalize(positive_boxes, *size)
    num_positives, num_negatives = compute_samples(positive_ratio, batch_size)
    negative_boxes = paz.boxes.sample_negatives(
        keys[0],
        positive_boxes,
        *size,
        box_size,
        num_negatives,
        num_negatives * 3,
    )

    positive_boxes = paz.boxes.sample_positives(
        keys[1],
        positive_boxes,
        *size,
        num_positives,
        scale_range,
        shift_range,
    )

    positive_boxes = paz.boxes.square(positive_boxes)
    negative_boxes = paz.boxes.square(negative_boxes)
    positive_images = paz.boxes.crop_with_pad(
        positive_boxes, image, *box_size, pad
    )
    positive_images = jax.vmap(augment)(
        jax.random.split(keys[2], len(positive_images)), positive_images
    )

    negative_images = paz.boxes.crop_with_pad(
        negative_boxes, image, *box_size, pad
    )
    negative_images = jax.vmap(augment)(
        jax.random.split(keys[3], len(negative_images)), negative_images
    )
    return build_labels(keys[4], positive_images, negative_images)
