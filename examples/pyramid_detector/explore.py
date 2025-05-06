from tqdm import tqdm
import jax
from deepfish import load
import jax.numpy as jp
import matplotlib.pyplot as plt
import paz
import plotter

train_images, train_labels = load("Deepfish/", "train")
valid_images, valid_labels = load("Deepfish/", "validation")
images = train_images + valid_images
labels = train_labels + valid_labels

boxes_per_image = jp.array([len(label) for label in labels])

# print(jp.unique(jp.array([paz.image.get_size(image)[0] for image in images])))

# dataset_H, dataset_W = jp.array([]), jp.array([])
dataset_H, dataset_W = [], []
for detections in tqdm(labels):
    boxes = paz.boxes.denormalize(
        paz.detection.get_boxes(detections), 1080, 1920
    )
    H, W = paz.boxes.compute_sizes(boxes, keepdims=False)
    H = H.tolist()
    W = W.tolist()
    dataset_H = dataset_H + H
    dataset_W = dataset_W + W
dataset_H = jp.array(dataset_H)
dataset_W = jp.array(dataset_W)

plotter.histogram(dataset_H, "Boxes Height")
plt.show()

plotter.histogram(dataset_W, "Boxes Width")
plt.show()

plotter.histogram_uniques(boxes_per_image, "Number of boxes")
plt.show()

key = jax.random.PRNGKey(777)
sample_arg = 0
detections = labels[sample_arg]
image = paz.image.load(images[sample_arg])
size = paz.image.get_size(image)
positive_boxes = paz.detection.get_boxes(detections)
positive_boxes = paz.boxes.denormalize(positive_boxes, *size)

image_with_boxes = paz.draw.boxes(image, positive_boxes)
paz.image.show(image_with_boxes)

negative_boxes = paz.boxes.sample_negatives(
    key, positive_boxes, *size, (100, 100), 100, 300
)

image_with_boxes = paz.draw.boxes(
    image_with_boxes, negative_boxes, color=(255, 0, 0)
)
paz.image.show(image_with_boxes)
