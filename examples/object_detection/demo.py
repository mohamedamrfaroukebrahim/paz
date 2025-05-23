import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
import paz
import jax.numpy as jp


def detect(
    image,
    model,
    prior_boxes,
    class_names,
    score_thresh,
    IOU_thresh,
    top_k,
    variances,
):
    image_size = model.input_shape[1:3]

    def preprocess(image, mean=paz.image.BGR_IMAGENET_MEAN):
        """Single-shot Multi Box Detector preprocessing function."""
        image = paz.image.resize(image, image_size, "linear", False)
        image = paz.image.RGB_to_BGR(image)
        image = paz.image.subtract_mean(image, jp.array(mean))
        image = paz.cast(image, "float32")
        image = jp.expand_dims(image, axis=0)
        return image

    def postprocess(detections):
        """Single-shot Multi Box Detector postprocessing function."""
        detections = jp.squeeze(detections, axis=0)
        detections = paz.detection.decode(detections, prior_boxes, variances)
        detections = paz.detection.remove_class(detections, 0)
        NMS_args = (len(class_names), IOU_thresh, top_k, 0.01)
        detections = paz.detection.apply_per_class_NMS(detections, *NMS_args)
        detections = paz.detection.filter_by_score(detections, score_thresh)
        return detections

    image = jax.jit(preprocess)(image)
    predictions = jax.jit(model)(image)
    # predictions = jax.jit(lambda x: model(preprocess(x)))(image)
    return jax.jit(postprocess, device=jax.devices("cpu")[0])(predictions)


def SSD300VOC(score_thresh=0.60, IOU_thresh=0.45, top_k=200):
    model = paz.models.detection.SSD300(21, "VOC", "VOC", (300, 300, 3))
    boxes = paz.models.detection.utils.create_prior_boxes("VOC")
    names = paz.datasets.labels("VOC")
    variances = [0.1, 0.1, 0.2, 0.2]
    args = (model, boxes, names, score_thresh, IOU_thresh, top_k, variances)
    return paz.lock(detect, *args)


def SSD512COCO(score_thresh=0.60, IOU_thresh=0.45, top_k=200):
    model = paz.models.detection.SSD512(81, "COCO", "COCO", (512, 512, 3))
    boxes = paz.models.detection.utils.create_prior_boxes("COCO")
    names = paz.datasets.labels("COCO")
    variances = [0.1, 0.1, 0.2, 0.2]
    args = (model, boxes, names, score_thresh, IOU_thresh, top_k, variances)
    return paz.lock(detect, *args)


if __name__ == "__main__":
    import jax

    image = paz.image.load("photo_2.jpg")
    # pipeline = SSD300VOC()
    pipeline = SSD512COCO()
    print(paz.log.time(pipeline, 20, 1, True, image))
    # detections = pipeline(image)
    # detections = paz.detection.remove_invalid(detections)
    # print(detections)

    # model = paz.models.detection.SSD300()
    # y = model(jp.ones((1, 300, 300, 3)))
