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
    keys = jax.random.split(key, 3)
    image = paz.image.random_flip_left_right(keys[0], image)
    image = paz.image.random_color_transform(keys[1], image)
    image = paz.image.random_rotation(keys[2], image, -jp.pi / 6, jp.pi / 6)
    return image


def sample_positives2(key, boxes):
    # shuffle boxes
    # do a for loop that iterates over the same boxes and jitters them.
    # square, clip
    return


def sample_positives(key, boxes, H, W, num_samples, scale_range, shift_range):

    def select_random_box(key, boxes):
        arg = jax.random.randint(key, shape=(), minval=0, maxval=len(boxes))
        return jp.expand_dims(boxes[arg], 0)

    def apply(boxes, key):
        box = select_random_box(key, boxes)
        box = paz.boxes.jitter(key, box, H, W, scale_range, shift_range)
        return boxes, jp.squeeze(box, axis=0)

    keys = jax.random.split(key, num_samples)
    _, jittered_boxes = jax.lax.scan(apply, boxes, keys)
    return jittered_boxes


def batch(
    key, image, detections, H_box, W_box, batch_size=32, positive_ratio=0.5
):
    # TODO change arguments to be positive ratio, batch_size
    num_positives = int(positive_ratio * batch_size)
    num_negatives = batch_size - num_positives
    keys = jax.random.split(key, 4)

    # box preprocesing
    H, W = paz.image.get_size(image)
    positive_boxes = paz.detection.get_boxes(detections)
    positive_boxes = paz.boxes.denormalize(positive_boxes, H, W)
    positive_boxes = sample_positives(
        key, positive_boxes, H, W, num_positives, (0.8, 1.1), (-10, 10)
    )
    positive_boxes = paz.boxes.square(positive_boxes)
    positive_boxes = paz.boxes.clip(positive_boxes, H, W)

    # positive preprocessing
    positive_images = crop_and_resize(image, positive_boxes, H_box, W_box)
    positive_keys = jax.random.split(keys[1], len(positive_images))
    positive_images = jax.vmap(augment)(positive_keys, positive_images)

    # negative preprocessing
    negative_args = (
        keys[0],
        positive_boxes,
        H,
        W,
        box_size,
        num_negatives,
        num_negatives * 2,
    )
    negative_boxes = paz.boxes.sample_negatives(*negative_args)
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
