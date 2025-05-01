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
    keys = jax.random.split(key)
    image = paz.image.random_color_transform(keys[0], image)
    image = paz.image.random_rotation(keys[1], image, -jp.pi / 2, jp.pi / 2)
    return image


def batch(key, image, detections, H_box, W_box):
    keys = jax.random.split(key, 4)

    # box preprocesing
    H, W = paz.image.get_size(image)
    positive_boxes = paz.detection.get_boxes(detections)
    positive_boxes = paz.boxes.denormalize(positive_boxes, H, W)
    positive_boxes = paz.boxes.square(positive_boxes)
    # TODO add clip boxes

    # positive preprocessing
    positive_images = crop_and_resize(image, positive_boxes, H_box, W_box)
    positive_keys = jax.random.split(keys[1], len(positive_images))
    positive_images = jax.vmap(augment)(positive_keys, positive_images)
    # TODO sample more positives and trim

    # negative preprocessing
    sample_negative_args = (keys[0], positive_boxes, H, W, box_size, 10, 20)
    negative_boxes = paz.boxes.sample_negatives(*sample_negative_args)
    negative_images = crop_and_resize(image, negative_boxes, H_box, W_box)
    negative_keys = jax.random.split(keys[3], len(negative_images))
    negative_images = jax.vmap(augment)(negative_keys, negative_images)

    # building labels
    positive_labels = jp.full(len(positive_boxes), 1.0)
    negative_labels = jp.full(len(positive_boxes), 0.0)
    images = (negative_images, positive_images)
    labels = (negative_labels, positive_labels)
    return images, labels


image = paz.image.load(train_images[sample_arg])
detections = train_labels[sample_arg]
paz.image.show(image)

images, labels = batch(key, image, detections, 100, 100)
paz.image.show(paz.draw.mosaic(images[0].astype("uint8")).astype("uint8"))
paz.image.show(paz.draw.mosaic(images[1].astype("uint8")).astype("uint8"))
