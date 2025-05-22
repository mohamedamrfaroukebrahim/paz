import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
import paz
import jax.numpy as jp


def preprocess_SSD(image, image_size, mean=paz.image.BGR_IMAGENET_MEAN):
    """Single-shot Multi Box Detector preprocessing function."""
    image = paz.image.resize(image, image_size, "linear", False)
    image = paz.image.RGB_to_BGR(image)
    image = paz.image.subtract_mean(image, jp.array(mean))
    image = paz.cast(image, "float32")
    image = jp.expand_dims(image, axis=0)
    return image


def postprocess_SSD(
    detections,
    prior_boxes,
    class_names,
    score_thresh,
    IOU_thresh,
    top_k,
    variances,
):
    """Single-shot Multi Box Detector postprocessing function."""
    detections = jp.squeeze(detections, axis=0)
    detections = paz.detection.decode(detections, prior_boxes, variances)
    detections = paz.detection.remove_class(detections, 0)
    NMS_args = (len(class_names), IOU_thresh, top_k)
    detections = paz.detection.apply_per_class_NMS(detections, *NMS_args)
    detections = paz.detection.filter_by_score(detections, score_thresh)
    return detections


def detect_SSD(
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
    image = jax.jit(paz.lock(preprocess_SSD, image_size))(image)
    predictions = model(image)
    return jax.jit(
        paz.lock(
            postprocess_SSD,
            prior_boxes,
            class_names,
            score_thresh,
            IOU_thresh,
            top_k,
            variances,
        )
    )(predictions)


def SSD300(score_thresh=0.60, IOU_thresh=0.45, top_k=100):
    model = paz.models.detection.SSD300()
    model.compile(jit_compile=True)
    prior_boxes = paz.models.detection.utils.create_prior_boxes("VOC")
    names = paz.datasets.labels("VOC")
    variances = [0.1, 0.1, 0.2, 0.2]
    return paz.lock(
        detect_SSD,
        model,
        prior_boxes,
        names,
        score_thresh,
        IOU_thresh,
        top_k,
        variances,
    )


def filter_invalid_boxes(detections_array):
    is_invalid_row_mask = jp.all(detections_array == -1, axis=1)
    is_valid_row_mask = jp.logical_not(is_invalid_row_mask)
    valid_boxes = detections_array[is_valid_row_mask]
    return valid_boxes


if __name__ == "__main__":
    import jax

    image = paz.image.load("photo_2.jpg")
    pipeline = SSD300()
    print(paz.log.time(pipeline, 20, 1, True, image))
    # detections = pipeline(image)
    # print(detections)
    # print(filter_invalid_boxes(detections))

    # model = paz.models.detection.SSD300()
    # y = model(jp.ones((1, 300, 300, 3)))
