import argparse
from paz.applications import HaarCascadeFaceDetector

parser = argparse.ArgumentParser(description="HaarCascadeDetector")
parser.add_argument(
    "-m",
    "--models",
    nargs="+",
    type=str,
    default=["frontalface_default", "eye"],
    help="Model name postfix of openCV xml file",
)
parser.add_argument(
    "-c", "--camera_id", type=int, default=0, help="Camera device ID"
)
args = parser.parse_args()


detect = HaarCascadeFaceDetector(1.3, 5, 0)

if args.image_path is None:
    camera = Camera(args.camera_id)
    player = VideoPlayer((640, 480), pipeline, camera)
    player.run()
else:
    image = load_image(args.image_path)
    predictions = pipeline(image)
    show_image(predictions["image"])
