import argparse
import paz
from paz.applications import HaarCascadeFaceDetector

# from paz import Camera, VideoPlayer

parser = argparse.ArgumentParser(description="HaarCascadeDetector")
parser.add_argument(
    "-m",
    "--models",
    nargs="+",
    type=str,
    default=["frontalface_default", "eye"],
    help="Model name postfix of openCV xml file",
)
parser.add_argument("--image_path", default=0, type=int)
parser.add_argument("--camera", default=0, type=int)
parser.add_argument("--H", default=480, type=int)
parser.add_argument("--W", default=640, type=int)
args = parser.parse_args()


pipeline = HaarCascadeFaceDetector(1.3, 5, 0)
camera = paz.Camera(args.camera)
player = paz.VideoPlayer((args.H, args.W), pipeline, camera)
player.run()
