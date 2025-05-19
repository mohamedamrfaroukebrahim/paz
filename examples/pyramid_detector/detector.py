import os

os.environ["KERAS_BACKEND"] = "jax"
import jax
import jax.numpy as jp

from deepfish import load
from patch import boxes_patch
import ensemble
import paz

H = 480
W = 640
patch_size = (128, 128)
strides = (32, 32)
score_threshold = 0.90
entropy_threshold = 0.75
iou_threshold = 0.1
num_models = 20
scales = [1.0]
padding = "same"
model = "simple"

valid_images, valid_labels = load("Deepfish/", "validation")
models = ensemble.load(f"experiments/*_ensemble_*/{model}.keras", num_models)
predict = jax.jit(paz.partial(ensemble.predict, models))
patch = jax.jit(paz.lock(paz.image.patch, patch_size, strides, padding))


for sample_arg in range(0, len(valid_images), 100):
    image = paz.image.load(valid_images[sample_arg])
    label = valid_labels[sample_arg]
    image = paz.image.resize(image, (H, W))
    pyramid_boxes, pyramid_score = [], []
    for pyramid, scale in zip(paz.image.pyramid(image, scales=scales), scales):
        patches = patch(pyramid)
        patches = patches.reshape(-1, *patch_size, 3)
        boxes = boxes_patch(
            *paz.image.get_size(pyramid), patch_size, strides, padding
        )
        boxes = boxes.reshape(-1, 4)
        model_scores = predict(patches)
        model_scores = jp.squeeze(model_scores, axis=-1)
        score = jp.mean(model_scores, axis=0)
        is_positive = score > score_threshold
        entropies = jax.vmap(ensemble.compute_entropy, in_axes=1)(model_scores)
        print(entropies)
        i_am_certain = entropies < entropy_threshold
        is_positive = is_positive & i_am_certain
        pyramid_boxes.append(paz.boxes.scale(boxes[is_positive], scale, scale))
        pyramid_score.append(score[is_positive])
    pyramid_boxes = jp.concatenate(pyramid_boxes, axis=0)
    pyramid_score = jp.concatenate(pyramid_score, axis=0)
    pyramid_boxes = jp.array(pyramid_boxes).reshape(-1, 4)
    if len(pyramid_boxes) == 0:
        print("No boxes found")
        paz.image.show(image.astype("uint8"))
    else:
        args = paz.boxes.apply_NMS(pyramid_boxes, pyramid_score, iou_threshold)
        positive_boxes = pyramid_boxes[args].astype(jp.int32)
        paz.image.show(paz.draw.boxes(image.astype("uint8"), positive_boxes))
