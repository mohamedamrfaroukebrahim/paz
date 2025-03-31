import os
import glob
import shutil

import jax.numpy as jp

import gdown

GDRIVE_LABEL = "10Pr4lLeSGTfkjA40ReGSC8H3a9onfMZ0"


def extract(filepath):
    output_path = os.path.dirname(filepath)
    shutil.unpack_archive(filepath, output_path)
    return output_path


def download(filename="deepfish.zip", gdrive_label=GDRIVE_LABEL):
    URL = f"https://drive.google.com/uc?export=download&id={gdrive_label}"
    gdown.download(URL, filename, quiet=False)
    return filename


def parse_line(line):
    values = [float(value) for value in line.strip().split()]
    # class_arg, x_min, y_min, x_max, y_max = values
    class_arg, x_center, y_center, box_W, box_H = values
    x_min = x_center - (box_W / 2)
    x_max = x_center + (box_W / 2)
    y_min = y_center - (box_H / 2)
    y_max = y_center + (box_H / 2)
    return x_min, y_min, x_max, y_max, class_arg


def parse(filepath):
    boxes = []
    with open(filepath, "r") as file:
        boxes.extend([parse_line(line) for line in file])
    return jp.array(boxes)


def load(path, split="train"):
    if split == "validation":
        split = "valid"
    images = sorted(glob.glob(f"{path}/*/{split}/*.jpg"))
    files = sorted(glob.glob(f"{path}/*/{split}/*.txt"))
    boxes = [parse(file) for file in files]
    return images, boxes


if __name__ == "__main__":
    import paz

    train_images, train_labels = load("Deepfish/", "train")
    valid_images, valid_labels = load("Deepfish/", "validation")
    print("Total num images", len(train_images) + len(valid_images))
    for path, detections in zip(train_images, train_labels):
        print(path)
        image = paz.image.load(path)
        image_boxes, class_args = paz.detection.split(detections)
        H, W = paz.image.get_dimensions(image)
        image_boxes = (image_boxes * jp.array([[W, H, W, H]])).astype(int)
        image = paz.draw.boxes(image, image_boxes)
        paz.image.show(image)
