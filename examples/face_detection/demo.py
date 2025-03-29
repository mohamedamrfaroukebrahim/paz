import argparse
import paz
import jax.numpy as jp
from paz.applications import HaarCascadeFaceDetector
from paz.applications import HaarCascadeEyeDetector

# from paz import Camera, VideoPlayer

parser = argparse.ArgumentParser(description="HaarCascadeDetector")
parser.add_argument("--image_path", default=0, type=int)
parser.add_argument("--camera", default=0, type=int)
parser.add_argument("--H", default=480, type=int)
parser.add_argument("--W", default=640, type=int)
args = parser.parse_args()


pipeline = HaarCascadeEyeDetector()
pipeline = HaarCascadeFaceDetector()


def FaceEyeDetector():
    detect_face = HaarCascadeFaceDetector()
    detect_eyes = HaarCascadeEyeDetector()

    def call(image):
        face_detections = detect_face(image)
        eyes_detections = detect_eyes(image)
        boxes = jp.concatenate(
            [face_detections.boxes, eyes_detections.boxes], axis=0
        )
        image = paz.draw.boxes(image, boxes)
        return paz.NamedTuple("State", image=image, boxes=boxes)

    return call


# image = paz.Input("image")
# x1 = paz.Node(HaarCascadeFaceDetector())(image)
# x2 = paz.Node(HaarCascadeEyeDetector())(image)
# paz.Node(lambda x: paz.draw.boxes(x.image, x.boxes, paz.draw.GREEN, 2))()
# paz.Model(image, [])
pipeline = FaceEyeDetector()
camera = paz.Camera(args.camera)
player = paz.VideoPlayer((args.H, args.W), pipeline, camera)
player.run()
