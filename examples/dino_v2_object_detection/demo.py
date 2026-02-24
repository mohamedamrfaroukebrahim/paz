import os
import numpy as np
from tensorflow.keras.utils import get_file
from paz.backend.image import load_image, show_image
from paz.models.detection.dino_v2 import DINOv2ViTS

URL = ('https://github.com/oarriaga/altamira-data/releases/download/v0.16/'
       'image_with_multiple_objects.png')

filename = os.path.basename(URL)
fullpath = get_file(filename, URL, cache_subdir='paz/tests')
image = load_image(fullpath)

model = DINOv2ViTS(num_classes=80)
print('DINOv2-ViTS model created. Input shape:', model.input_shape)
print('DINOv2-ViTS model output shape:', model.output_shape)
show_image(image)
