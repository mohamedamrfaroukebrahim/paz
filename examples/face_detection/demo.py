import argparse
import paz
from paz.applications import HaarCascadeFaceDetector
from paz.applications import HaarCascadeEyeDetector

# from paz import Camera, VideoPlayer

parser = argparse.ArgumentParser(description="HaarCascadeDetector")
parser.add_argument("--image_path", default=0, type=int)
parser.add_argument("--camera", default=0, type=int)
parser.add_argument("--H", default=480, type=int)
parser.add_argument("--W", default=640, type=int)
args = parser.parse_args()


pipeline = HaarCascadeFaceDetector()
pipeline = HaarCascadeEyeDetector()
camera = paz.Camera(args.camera)
player = paz.VideoPlayer((args.H, args.W), pipeline, camera)
player.run()
