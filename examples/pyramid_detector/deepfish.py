import os
import gdown
import shutil

GDRIVE_LABEL = "10Pr4lLeSGTfkjA40ReGSC8H3a9onfMZ0"


def extract(filepath):
    output_path = os.path.dirname(filepath)
    shutil.unpack_archive(filepath, output_path)
    return output_path


def download(filename="deepfish.zip", gdrive_label=GDRIVE_LABEL):
    URL = f"https://drive.google.com/uc?export=download&id={gdrive_label}"
    gdown.download(URL, filename, quiet=False)
    return filename


# extract(download())
