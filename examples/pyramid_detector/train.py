# scale the box to not be exactly always over the object
# Training pipeline
# sample n negatives
# sample m positives
# apply color jitter augmentation
# apply image transformation augmentation
# clip
# crop
# resize
# return label and batch
# train classifier
# fine tune convnext
# train mini-xception
# build image pyramid detection pipeline
# build ensemble
# model = paz.models.MiniXception((*box_size, 3), 1)
# model.summary()
# TODO add this
# positive_boxes = sample(positive_boxes, num_positives)
# I can also sample positive samples.
# Thus, I can also generate a large amount of positives and negatives.
# Thus I can make batches.
import os

os.environ["KERAS_BACKEND"] = "jax"
import jax
import jax.numpy as jp
import paz
from deepfish import load
import matplotlib.pyplot as plt
import numpy as np
import math

train_images, train_labels = load("Deepfish/", "train")
valid_images, valid_labels = load("Deepfish/", "validation")
box_size = (128, 128)
key = jax.random.PRNGKey(777)
sample_arg = 0


def crop_and_resize(image, boxes, H_box, W_box):
    def apply(box):
        return paz.image.resize(paz.image.crop(image, box), (H_box, W_box))

    return jp.array([apply(box) for box in boxes])


def augment(key, image):
    key_0, key_1, key_2, key_3, key_4, key_5 = jax.random.split(key, 6)
    image = paz.image.random_flip_left_right(key_0, image)
    # image = paz.image.random_saturation(key_1, image)
    # image = paz.image.random_brightness(key_2, image)
    # image = paz.image.random_contrast(key_3, image)
    image = paz.image.random_hue(key_4, image, max_delta=0.0001)
    # image = paz.image.random_rotation(key_5, image, -jp.pi / 6, jp.pi / 6)
    return image


def compute_samples(positive_ratio, batch_size):
    num_positives = int(positive_ratio * batch_size)
    num_negatives = batch_size - num_positives
    return num_positives, num_negatives


def preprocess_boxes(boxes, H, W):
    boxes = paz.detection.get_boxes(detections)
    boxes = paz.boxes.denormalize(boxes, H, W)
    return boxes


def build_positives(key, boxes, H, W, num_samples, scale_range, shift_range):
    # sample_args = (key, boxes, H, W, num_samples, (0.8, 1.1), (-10, 10))
    sample_args = (key, boxes, H, W, num_samples, scale_range, shift_range)
    positive_boxes = paz.boxes.sample_positives(*sample_args)
    positive_boxes = paz.boxes.square(boxes)
    positive_boxes = paz.boxes.clip(positive_boxes, H, W)
    return positive_boxes


def build_labels(
    key, positive_images, negative_images, positive_boxes, negative_boxes
):
    positive_labels = jp.full(len(positive_boxes), 1.0)
    negative_labels = jp.full(len(positive_boxes), 0.0)
    images = jp.concatenate([negative_images, positive_images], axis=0)
    labels = jp.concatenate([negative_labels, positive_labels], axis=0)
    shuffled_args = jax.random.permutation(key, jp.arange(len(images)))
    images = images[shuffled_args]
    labels = labels[shuffled_args]
    return images, labels


def to_images(key, image, boxes, H_box, W_box):
    images = crop_and_resize(image, boxes, H_box, W_box)
    images = jax.vmap(augment)(jax.random.split(key, len(images)), images)
    return images


def display_images_with_labels(images, labels):
    num_images = images.shape[0]
    cols = min(int(math.ceil(math.sqrt(num_images))), 8)
    rows = int(math.ceil(num_images / cols))

    figure, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    for image_arg in range(num_images):
        axes[image_arg].imshow(images[image_arg])
        axes[image_arg].set_title(f"{labels[image_arg]}")
        axes[image_arg].axis("off")

    for image_arg in range(num_images, len(axes)):
        axes[image_arg].axis("off")

    plt.tight_layout()
    return figure


def batch(
    key,
    image,
    detections,
    H_box,
    W_box,
    batch_size=32,
    positive_ratio=0.5,
    scale_range=(1.3, 1.8),
    shift_range=(0, 0),
):
    keys = jax.random.split(key, 5)
    num_positives, num_negatives = compute_samples(positive_ratio, batch_size)
    size = paz.image.get_size(image)
    boxes = preprocess_boxes(detections, *size)
    args = (keys[0], boxes, *size, num_positives, scale_range, shift_range)
    positive_boxes = build_positives(*args)
    positive_images = to_images(keys[1], image, positive_boxes, H_box, W_box)

    num_trials = num_negatives * 2
    args = (keys[2], positive_boxes, *size, box_size, num_negatives, num_trials)
    negative_boxes = paz.boxes.sample_negatives(*args)
    negative_images = to_images(keys[3], image, negative_boxes, H_box, W_box)
    return build_labels(
        keys[4],
        positive_images,
        negative_images,
        positive_boxes,
        negative_boxes,
    )


image = paz.image.load(train_images[sample_arg])
detections = train_labels[sample_arg]

# show positives
paz.image.show(
    paz.draw.boxes(
        image,
        paz.boxes.denormalize(
            paz.detection.get_boxes(detections), *paz.image.get_size(image)
        ),
    )
)


images, labels = batch(key, image, detections, 100, 100)
paz.image.show(paz.draw.mosaic(images.astype("uint8")).astype("uint8"))
display_images_with_labels(images, labels)
plt.show()
