import jax
import jax.numpy as jp
import paz
from deepfish import load

train_images, train_labels = load("Deepfish/", "train")
valid_images, valid_labels = load("Deepfish/", "validation")
# print("Total num images", len(train_images) + len(valid_images))
# for path, detections in zip(train_images, train_labels):
#     image = paz.image.load(path)
#     image_boxes, class_args = paz.detection.split(detections)
#     H, W = paz.image.get_dimensions(image)
#     image_boxes = (image_boxes * jp.array([[W, H, W, H]])).astype(int)
#     image_boxes = paz.boxes.square(image_boxes)
#     image = paz.draw.boxes(image, image_boxes)
#     paz.image.show(image)


image = paz.image.load(train_images[0])
image_boxes, class_args = paz.detection.split(train_labels[0])
H, W = paz.image.get_dimensions(image)
image_boxes = (image_boxes * jp.array([[W, H, W, H]])).astype(int)
image_boxes = paz.boxes.square(image_boxes)
image_with_boxes = paz.draw.boxes(image, image_boxes)
paz.image.show(image_with_boxes)

num_trials = 5
H_box, W_box = 128, 128
key = jax.random.key(777)
keys = jax.random.split(key)
H, W = paz.backend.image.get_dimensions(image)

x_min = jax.random.randint(keys[0], (num_trials, 1), 0, W)
y_min = jax.random.randint(keys[1], (num_trials, 1), 0, H)
x_max = x_min + W_box
y_max = y_min + H_box
random_boxes = paz.boxes.merge(x_min, y_min, x_max, y_max)
ious = paz.boxes.compute_IOUs(random_boxes, image_boxes)
mean_iou = jp.mean(ious, axis=1)
best_arg = jp.argmin(mean_iou)
best_box = random_boxes[best_arg]

# random_box = jp.array([[x_min, y_min, x_max, y_max]])
# image = paz.draw.boxes(image, random_box)
# paz.image.show(image)
