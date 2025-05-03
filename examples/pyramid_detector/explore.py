from deepfish import load
import jax.numpy as jp
import matplotlib.pyplot as plt
import plotter

train_images, train_labels = load("Deepfish/", "train")
valid_images, valid_labels = load("Deepfish/", "validation")
images = train_images + valid_images
labels = train_labels + valid_labels

boxes_per_image = []
for label in labels:
    boxes_per_image.append(len(label))
boxes_per_image = jp.array(boxes_per_image)
plotter.histogram(
    boxes_per_image,
    title="Boxes Distribution",
    xlabel="Number of boxes",
)
# plt.hist(boxes_per_image)
plt.show()
