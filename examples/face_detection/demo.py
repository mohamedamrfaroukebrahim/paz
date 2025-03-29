import argparse
import paz
from paz.models.detection import HaarCascadeDetector

parser = argparse.ArgumentParser(description="HaarCascadeDetector")
parser.add_argument("--image_path", default=0, type=int)
parser.add_argument("--camera", default=0, type=int)
parser.add_argument("--H", default=480, type=int)
parser.add_argument("--W", default=640, type=int)
parser.add_argument("--models", nargs=2, default=["frontaface_default", "eye"])
args = parser.parse_args()


def Detector(labels, colors):
    draw_functions = [paz.lock(paz.draw.boxes, color, 2) for color in colors]

    detectors = []
    for class_arg, (label, draw) in enumerate(zip(labels, draw_functions)):
        detectors.append(HaarCascadeDetector(label, 1.3, 5, class_arg, draw))

    def call(image):
        boxes = paz.boxes.join([detect(image).boxes for detect in detectors])
        return paz.message.Detections(image, boxes)

    return call


colors = paz.draw.lincolor(len(args.models), normalize=False)
pipeline = Detector(args.models, colors)
camera = paz.Camera(args.camera)
player = paz.VideoPlayer((args.H, args.W), pipeline, camera)
player.run()
