import os
import shutil
import keras
import re
from typing import Optional
from urllib.parse import urlparse
from tqdm import tqdm
import gdown


def get_image_files(src_folder):
    """
    Returns a list of image files in the source folder.
    """
    image_files = []
    for image_file in os.listdir(src_folder):
        if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_files.append(image_file)
    return image_files


def get_file_paths(
    src_folder: str, dst_img_folder: str, dst_label_folder: str, img_file: str
):
    """
    Returns the source and destination paths for an image and its corresponding label file.
    """
    src_img_path = os.path.join(src_folder, img_file)
    dst_img_path = os.path.join(dst_img_folder, img_file)
    label_file = img_file.rsplit(".", 1)[0] + ".txt"
    src_label_path = os.path.join(src_folder, label_file)
    dst_label_path = os.path.join(dst_label_folder, label_file)
    return src_img_path, dst_img_path, src_label_path, dst_label_path


def copy_label_file(src_label_path: str, dst_label_path: str, img_file: str):
    """Copies the label file if it exists; otherwise, warns the user."""
    if os.path.exists(src_label_path):
        shutil.copy(src_label_path, dst_label_path)
    else:
        print(f"Warning: No label found for image {img_file}")


def process_folder(src_folder, dst_img_folder, dst_label_folder):
    """
    Copies images and their corresponding label files
    from the source folder to the destination folders.

    Args:
        src_folder (str): Path to the source folder containing images and labels.
        dst_img_folder (str): Path to the destination folder for images.
        dst_label_folder (str): Path to the destination folder for labels.
    """
    image_files = get_image_files(src_folder)
    for img_file in image_files:
        src_img_path, dst_img_path, src_label_path, dst_label_path = (
            get_file_paths(
                src_folder, dst_img_folder, dst_label_folder, img_file
            )
        )
        shutil.copy(src_img_path, dst_img_path)
        copy_label_file(src_label_path, dst_label_path, img_file)


def create_output_directories(output_path):
    """
    Creates necessary directories for images and labels.

    Args:
        output_path (str): Path to the output directory.
    """
    for split in ["train", "valid"]:
        os.makedirs(os.path.join(output_path, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_path, "labels", split), exist_ok=True)


def write_classes_file(output_path):
    """
    Writes the class names into classes.txt.

    Args:
        output_path (str): Path to the output directory.
    """
    with open(os.path.join(output_path, "classes.txt"), "w") as f:
        f.write("Fish\n")  # Only one class: Fish


def process_class_folders(raw_data_path, output_path):
    """
    Processes class folders (named as digits) for both 'train' and 'valid' splits.

    Args:
        raw_data_path (str): Path to the raw data directory containing class folders.
        output_path (str): Path to the output directory.
    """
    for class_folder in tqdm(os.scandir(raw_data_path), desc="Class Folders"):
        if class_folder.is_dir() and class_folder.name.isdigit():
            for split in ["train", "valid"]:
                src_folder = os.path.join(class_folder.path, split)
                dst_img_folder = os.path.join(output_path, "images", split)
                dst_label_folder = os.path.join(output_path, "labels", split)
                if os.path.exists(src_folder):
                    process_folder(
                        src_folder, dst_img_folder, dst_label_folder
                    )


def process_negative_samples(raw_data_path, output_path):
    """
    Processes negative sample images for both 'train' and 'valid' splits.

    Args:
        raw_data_path (str): Path to the raw data directory containing negative samples.
        output_path (str): Path to the output directory.
    """
    negative_samples_path = os.path.join(raw_data_path, "Negative_samples")
    for split in ["train", "valid"]:
        src_folder = os.path.join(negative_samples_path, split)
        dst_img_folder = os.path.join(output_path, "images", split)
        dst_label_folder = os.path.join(output_path, "labels", split)
        if os.path.exists(src_folder):
            process_folder(src_folder, dst_img_folder, dst_label_folder)


def process_class_data(raw_data_path, output_path):
    """
    Orchestrates the entire data processing workflow.

    Args:
        raw_data_path (str): Path to the raw data directory.
        output_path (str): Path to the output directory.
    """
    create_output_directories(output_path)
    write_classes_file(output_path)
    process_class_folders(raw_data_path, output_path)
    process_negative_samples(raw_data_path, output_path)


def get_gdrive_file_id(gdrive_link: str):
    """
    Extracts the file ID from a Google Drive link.

    Args:
        gdrive_link (str): Google Drive file URL.

    Returns:
        str or None: Extracted file ID or None if invalid.
    """
    pattern = r"(?:/d/|id=)([a-zA-Z0-9_-]+)"
    match = re.search(pattern, gdrive_link)
    gdrive_file_id = match.group(1) if match else None
    if not gdrive_file_id:
        raise ValueError(
            "Invalid Google Drive link: Could not extract file ID."
        )
    return gdrive_file_id


def download_from_google_drive(file_url: str, output_path: str) -> str:
    """
    Downloads a file from Google Drive given its URL and output path.

    Args:
        file_url (str): The original Google Drive file URL.
        output_path (str): The local path where the file will be saved.

    Returns:
        str: The output path if the download is successful.

    Raises:
        RuntimeError: If the download fails.
    """
    gdrive_file_id = get_gdrive_file_id(file_url)
    url = f"https://drive.google.com/uc?export=download&id={gdrive_file_id}"
    print(f"Downloading from Google Drive: {file_url}")
    try:
        gdown.download(url, output_path, quiet=False)
        print(f"Downloaded file: {output_path}")
        return output_path
    except Exception as e:
        raise RuntimeError(f"Failed to download from Google Drive: {e}")


def download_from_regular_url(
    file_url: str, output_filename: str, output_dir: Optional[str] = None
) -> str:
    """
    Downloads a file from a regular URL using keras.utils.get_file.

    Args:
        file_url (str): The URL of the file.
        output_filename (str): The filename to save the file as.
        output_dir (str): The directory where the file should be cached/downloaded.

    Returns:
        str: Local file path of the downloaded file.

    Raises:
        RuntimeError: If the download fails.
    """
    try:
        downloaded_file_local_path = keras.utils.get_file(
            fname=output_filename, origin=file_url, cache_dir=output_dir
        )
        print(f"Downloaded file: {downloaded_file_local_path}")
        return downloaded_file_local_path
    except Exception as e:
        raise RuntimeError(f"Failed to download file from URL: {e}")


def is_valid_url(url: str) -> bool:
    """
    Validate a URL by checking its scheme and netloc.
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def make_output_dir(
    output_filename: str, output_dir: Optional[str] = None
) -> str:
    """
    Creates an output directory if it does not already exist
    and returns the full path to the output file.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.abspath(""), "downloads")

    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, output_filename)


def get_file(
    file_url: str, output_filename: str, output_dir: Optional[str] = None
) -> str:
    """
    Downloads a file from a URL or Google Drive.

    Args:
        file_url (str): The URL of the file to download.
        output_filename (str): The name to save the file as.
        output_dir (str, optional): The directory to save the file
                                    (default: ~/.keras/).

    Returns:
        str: The local file path.
    """
    output_path = make_output_dir(output_filename, output_dir)

    if "drive.google.com" in urlparse(file_url).netloc:
        return download_from_google_drive(file_url, output_path)
    else:
        return download_from_regular_url(file_url, output_filename, output_dir)


def extract_compresed_file(file_path: str, output_dir: Optional[str] = None):
    """
    Extracts a compressed file (.zip, .tar, .rar, etc.).

    Args:
        file_path (str): Path to the compressed file.
        output_dir (str, optional): Directory to extract to
                                    (default: same as file location).

    Returns:
        str: Path to the extracted folder.
    """

    if output_dir is None:
        output_dir = os.path.dirname(file_path)

    try:
        shutil.unpack_archive(file_path, output_dir)
        print(f"Extracted to: {output_dir}")
        return output_dir
    except Exception as e:
        raise RuntimeError(f"Failed to extract file: {e}")


if __name__ == "__main__":
    downloaded_file_local_path = get_file(
        file_url="https://drive.google.com/file/d/10Pr4lLeSGTfkjA40ReGSC8H3a9onfMZ0/view",
        output_filename="fish_dataset_.zip",
    )
    extract_compresed_file(
        downloaded_file_local_path,
    )
    raw_data_path = r"Data/raw/Deepfish"
    output_path = r"Data/processed/yolo_dataset"
    process_class_data(
        raw_data_path=os.path.join(
            os.path.dirname(downloaded_file_local_path), "Deepfish"
        ),
        output_path=r"Data/processed/yolo_dataset",
    )
