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

num_trials = 15
num_boxes = 5
H_box, W_box = 128, 128
key = jax.random.key(777)
keys = jax.random.split(key)
# get shape / get_size
H, W = paz.image.get_dimensions(image)

x_min = jax.random.randint(keys[0], (num_trials, 1), 0, W)
y_min = jax.random.randint(keys[1], (num_trials, 1), 0, H)
x_max = x_min + W_box
y_max = y_min + H_box
random_boxes = paz.boxes.merge(x_min, y_min, x_max, y_max)
ious = paz.boxes.compute_IOUs(random_boxes, image_boxes)
mean_ious = jp.mean(ious, axis=1)
best_args = jp.argsort(mean_ious)[::-1]
best_args = best_args[:num_boxes]
best_boxes = random_boxes[best_args]
image_with_boxes = paz.draw.boxes(image_with_boxes, best_boxes, paz.draw.RED)
paz.image.show(image_with_boxes)

# random_box = jp.array([[x_min, y_min, x_max, y_max]])
# image = paz.draw.boxes(image, random_box)
# paz.image.show(image)
# keys = jax.random.split(jax.random.key(777), 3)
# image = paz.image.random_saturation(keys[0], image)
# paz.image.show(image)
# image = paz.image.random_brightness(keys[1], image)
# paz.image.show(image)
# image = paz.image.random_contrast(keys[2], image)
# paz.image.show(image)

# for key in jax.random.split(key, 10):
#     paz.image.show(paz.image.random_saturation(key, image.copy(), 0.5, 2.0))

# for key in jax.random.split(key, 10):
#     paz.image.show(paz.image.random_brightness(key, image.copy(), 100))

# for key in jax.random.split(key, 100):
#     paz.image.show(paz.image.random_contrast(key, image.copy()))


for key in jax.random.split(key, 20):
    keys = jax.random.split(key, 3)
    img = paz.image.random_saturation(keys[0], image.copy())
    img = paz.image.random_brightness(keys[1], img)
    img = paz.image.random_contrast(keys[2], img)
    paz.image.show(img)
