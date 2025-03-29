import os

os.environ["KERAS_BACKEND"] = "jax"

import jax.numpy as jp
import paz
from paz.applications import MiniXceptionFER

# model = MiniXception()


def to_labels(probabilities, labels):
    return labels[jp.argmax(probabilities)]


def preprocess(image, shape):
    image = paz.image.resize(image, shape)
    image = paz.image.normalize(image)
    image = paz.image.rgb_to_gray(image)
    return jp.expand_dims(image, [0, -1])


def ClassifyMiniXceptionFER():
    x = image = paz.Input("image")
    x = paz.Node(preprocess, (48, 48))(x)
    x = paz.Node(MiniXceptionFER(), name="score")(x)
    x = paz.Node(to_labels, paz.datasets.labels("FER"), name="label")(x)
    return paz.Model([image], [x], "MiniXceptionFER")


# score = MiniXception((48, 48, 1), 7, weights="FER")
classify = ClassifyMiniXceptionFER()
y = classify(jp.full((128, 128, 3), 255))


class DetectMiniXceptionFER(Processor):
    """Emotion classification and detection pipeline.

    # Returns
        Dictionary with ``image`` and ``boxes2D``.

    # Example
        ``` python
        from paz.pipelines import DetectMiniXceptionFER

        detect = DetectMiniXceptionFER()

        # apply directly to an image (numpy-array)
        inferences = detect(image)
        ```
    # Returns
        A function that takes an RGB image and outputs the predictions
        as a dictionary with ``keys``: ``image`` and ``boxes2D``.
        The corresponding values of these keys contain the image with the drawn
        inferences and a list of ``paz.abstract.messages.Boxes2D``.

    # References
       - [Real-time Convolutional Neural Networks for Emotion and
            Gender Classification](https://arxiv.org/abs/1710.07557)
    """

    def __init__(self, offsets=[0, 0], colors=EMOTION_COLORS):
        super(DetectMiniXceptionFER, self).__init__()
        self.offsets = offsets
        self.colors = colors

        # detection
        self.detect = HaarCascadeFrontalFace()
        self.square = SequentialProcessor()
        self.square.add(pr.SquareBoxes2D())
        self.square.add(pr.OffsetBoxes2D(offsets))
        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()

        # classification
        self.classify = MiniXceptionFER()

        # drawing and wrapping
        self.class_names = self.classify.class_names
        self.draw = pr.DrawBoxes2D(self.class_names, self.colors, True)
        self.wrap = pr.WrapOutput(["image", "boxes2D"])

    def call(self, image):
        boxes2D = self.detect(image.copy())["boxes2D"]
        boxes2D = self.square(boxes2D)
        boxes2D = self.clip(image, boxes2D)
        cropped_images = self.crop(image, boxes2D)
        for cropped_image, box2D in zip(cropped_images, boxes2D):
            predictions = self.classify(cropped_image)
            box2D.class_name = predictions["class_name"]
            box2D.score = np.amax(predictions["scores"])
        image = self.draw(image, boxes2D)
        return self.wrap(image, boxes2D)
