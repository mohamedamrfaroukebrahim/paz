import os
from paz.backend.image import load
import numpy as np


def get_image_extensions():
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    return image_extensions


def validate_directories(images_directory: str, labels_directory: str):
    """Validates that the provided directories exist."""
    if not os.path.isdir(images_directory):
        raise FileNotFoundError(
            f"Images directory does not exist: {images_directory}"
        )
    if not os.path.isdir(labels_directory):
        raise FileNotFoundError(
            f"Labels directory does not exist: {labels_directory}"
        )


def get_bases(file_paths: list):
    return [os.path.splitext(file_path)[0] for file_path in file_paths]


def get_missing_images(image_base_names, label_base_names):
    missing_images = []
    for base_name in label_base_names:
        if base_name not in image_base_names:
            missing_images.append(base_name)

    return missing_images


def get_missing_labels(image_base_names, label_base_names):
    missing_labels = []
    for base_name in image_base_names:
        if base_name not in label_base_names:
            missing_labels.append(base_name + ".txt")

    return missing_labels


def validate_image_correspondence(image_bases: list, label_bases: list):
    missing_images = get_missing_images(image_bases, label_bases)
    if missing_images:
        raise ValueError(
            "The following label files have no corresponding image files: "
            + str([base_name + ".[ext]" for base_name in missing_images])
        )


def validate_label_correspondence(image_bases: list, label_bases: list):
    missing_labels = get_missing_labels(image_bases, label_bases)
    if missing_labels:
        raise ValueError(
            "The following image files have no corresponding label files: "
            + str(missing_labels)
        )


def validate_file_correspondence(image_files: list, label_files: list):
    """
    Validates that each image file has a corresponding label file and vice versa.
    """
    image_bases, label_bases = get_bases(image_files), get_bases(label_files)
    validate_image_correspondence(image_bases, label_bases)
    validate_label_correspondence(image_bases, label_bases)


def _ensure_files_exist(files: list, directory: str, file_type: str) -> list:
    """
    Raise FileNotFoundError if files list is empty;
    otherwise return files.
    """
    if not files:
        raise FileNotFoundError(
            f"No {file_type} files found in the specified directory: {directory}"
        )
    return files


def get_image_files(images_directory: str):
    """
    Returns a list of image files from the images_directory
    matching the valid extensions.
    """
    image_files = []
    for image_file in os.listdir(images_directory):
        image_file_lower = image_file.lower()
        image_extensions = get_image_extensions()
        if image_file_lower.endswith(image_extensions):
            image_files.append(image_file)

    return _ensure_files_exist(image_files, images_directory, "image")


def get_label_files(labels_directory: str):
    """Returns a list of label files with a
    .txt extension from the labels_directory."""
    label_files = []
    for label_file in os.listdir(labels_directory):
        if label_file.endswith(".txt"):
            label_files.append(label_file)

    return _ensure_files_exist(label_files, labels_directory, "label")


def get_image_size(image_path: str):
    """
    Returns the size of the image at the given path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: (width, height) of the image.
    """
    image = load(image_path)
    if image is None:
        raise RuntimeError(f"Failed to load image: {image_path}")
    height, width, _ = image.shape
    return width, height


def parse_label_line(line: str) -> list:
    """
    Converts a line of text from the label file into a list of floats.
    """
    numbers = []
    for number in line.strip().split():
        numbers.append(float(number))

    return numbers


def load_label(label_path: str):
    """
    Loads labels from a file.

    Args:
        label_path (str): Path to the label file.

    Returns:
        np.ndarray: Array of labels, where each label is a list of floats.
    """
    labels = []
    with open(label_path, "r") as file:
        for line in file:
            labels.append(parse_label_line(line))
    return np.array(labels)


def process_normalized_labels(labels: np.ndarray):
    """
    Processes normalized YOLO format labels (0-1 range).

    Args:
        labels (np.ndarray): Array of labels in format
                           [class_id, x_center, y_center, width, height].

    Returns:
        np.ndarray: Array of labels in format [x_min, y_min, x_max, y_max, class_id].
    """
    processed_labels = []
    for label in labels:
        class_id, x_center, y_center, box_width, box_height = label
        x_min = x_center - box_width / 2
        x_max = x_center + box_width / 2
        y_min = y_center - box_height / 2
        y_max = y_center + box_height / 2
        processed_labels.append([x_min, y_min, x_max, y_max, int(class_id)])
    return np.array(processed_labels)


def process_absolute_labels(labels: np.ndarray, image_size: tuple):
    """
    Processes YOLO format labels and converts to absolute pixel coordinates.

    Args:
        labels (np.ndarray): Array of labels in format
                           [class_id, x_center, y_center, width, height].
        image_size (tuple): The (width, height) of the image.

    Returns:
        np.ndarray: Array of labels in format [x_min, y_min, x_max, y_max, class_id].
    """
    processed_labels = []
    width, height = image_size
    for label in labels:
        class_id, x_center, y_center, box_width, box_height = label
        x_min = int((x_center - box_width / 2) * width)
        y_min = int((y_center - box_height / 2) * height)
        x_max = int((x_center + box_width / 2) * width)
        y_max = int((y_center + box_height / 2) * height)
        processed_labels.append([x_min, y_min, x_max, y_max, int(class_id)])
    return np.array(processed_labels)


def process_labels(
    labels: np.ndarray, image_size: tuple, normalize: bool = True
):
    """
    Processes labels by converting coordinates to the desired format.

    Args:
        labels (np.ndarray): Array of labels in YOLO format.
        image_size (tuple): The (width, height) of the image.
        normalize (bool): If True, keeps coordinates normalized (0-1 range).
                        If False, converts to absolute pixel coordinates.

    Returns:
        np.ndarray: Array of processed labels.
    """
    if normalize:
        return process_normalized_labels(labels)
    return process_absolute_labels(labels, image_size)


def get_data_PAZ_format(
    images_directory: str, labels_directory: str, normalize: bool = True
):
    """
    Loads and processes image and label data
    for object detection in PAZ format.

    Args:
        images_directory (str): Path to the directory containing image files.
        labels_directory (str): Path to the directory containing label files.
        normalize (bool): Whether to normalize label coordinates.

    Returns:
        list: A list of dictionaries, each containing:
            - 'image' (str): Path to the image file.
            - 'boxes' (np.ndarray): Processed label data for the image.
    """
    data = []
    validate_directories(images_directory, labels_directory)
    image_files = get_image_files(images_directory)
    label_files = get_label_files(labels_directory)
    validate_file_correspondence(image_files, label_files)
    for file in image_files:
        image_path = os.path.join(images_directory, file)
        image_size = get_image_size(image_path)
        # Get the base name (without extension) and build the label file name
        base_name = os.path.splitext(file)[0]
        label_path = os.path.join(labels_directory, base_name + ".txt")

        labels = load_label(label_path)
        box_data = process_labels(labels, image_size, normalize=normalize)

        data.append({"image": image_path, "boxes": box_data})
    return data


if __name__ == "__main__":
    images_directory_full_path = r"./Data/processed/yolo_dataset/images/train"
    labels_directory_full_path = r"./Data/processed/yolo_dataset/labels/train"

    data = get_data_PAZ_format(
        images_directory_full_path, labels_directory_full_path, normalize=False
    )
    print(f"Number of data entries: {len(data)}")
    print(f"First data entry: {data[0]}")

    print("First data entry image path: ", data[0]["image"])
    print("First data entry box data: ", data[0]["boxes"])
