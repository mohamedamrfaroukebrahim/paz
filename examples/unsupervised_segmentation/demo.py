import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["KERAS_BACKEND"] = "jax"

import argparse
import paz
import numpy as np
import matplotlib.pyplot as plt
import torch

from model import (
    preprocess,
    compute_foreground_masks,
    compute_joint_features,
    project_features,
)


parser = argparse.ArgumentParser(description="Unsupervised Segmentation")
parser.add_argument("--images_path", default="data/hippogriff", type=str)
parser.add_argument("--H_crop", default=518, type=int)
parser.add_argument("--W_crop", default=518, type=int)
parser.add_argument("--threshold", default=0.35, type=float)
parser.add_argument("--mean", default=paz.image.rgb_IMAGENET_MEAN)
parser.add_argument("--stdv", default=paz.image.rgb_IMAGENET_STDV)
args = parser.parse_args()
crop_shape = (args.H_crop, args.W_crop)


def validate_crop_shape(crop_shape, patch_size):
    assert crop_shape[0] % patch_size == 0
    assert crop_shape[1] % patch_size == 0


images, grid_images = [], []
for filename in sorted(os.listdir(args.images_path)):
    image = paz.image.load(os.path.join(args.images_path, filename))
    images.append(image)
    grid_images.append(preprocess(image, crop_shape, 0, 1))
mosaic = paz.draw.mosaic(np.array(grid_images), border=5, background=0.0)
paz.image.show(paz.image.denormalize(mosaic))


model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
model.to(torch.device("cuda"))
patch_size = model.patch_size
validate_crop_shape(crop_shape, model.patch_size)


model_args = (model, crop_shape, patch_size, args.mean, args.stdv)
masks = compute_foreground_masks(*model_args, args.threshold, images)
joint_features = compute_joint_features(
    model, images, args.mean, args.stdv, crop_shape, patch_size
)
features = project_features(3, masks, joint_features)

plt.imshow(paz.draw.mosaic(masks, (2, 2), 2, 0))
plt.show()

plt.imshow(paz.draw.mosaic(features, (2, 2), 2, 0.0))
plt.show()
