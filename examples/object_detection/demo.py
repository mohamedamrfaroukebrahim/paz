import paz
import jax.numpy as jp


def SSDPreprocess(image, image_shape, mean=paz.image.BGR_IMAGENET_MEAN):
    """Single-shot Multi Box Detector preprocessing function."""
    image = paz.image.resize(image, image_shape)
    image = paz.image.RGB_to_BGR(image)
    image = paz.image.subtract_mean(image, mean)
    image = paz.image.cast(image, "float32")
    image = jp.expand_dims(image, axis=0)
    return image


def remove_class(detections, class_arg):
    """Remove a particular class from the pipeline.

    # Arguments
        class_names: List, indicating given class names.
        class_arg: Int, index of the class to be removed.
    """
    # del class_names[class_arg]
    return jp.delete(detections, 4 + class_arg, axis=1)


def SSDPostprocess(
    detections,
    model,
    class_names,
    score_thresh,
    nms_thresh,
    variances=[0.1, 0.1, 0.2, 0.2],
    class_arg=0,
    box_method=0,
):
    """Single-shot Multi Box Detector postprocessing function."""
    # self.add(pr.Squeeze(axis=None))
    # self.add(pr.DecodeBoxes(model.prior_boxes, variances))
    # self.add(pr.RemoveClass(class_names, class_arg, renormalize=False))
    # self.add(pr.NonMaximumSuppressionPerClass(nms_thresh))
    # self.add(pr.MergeNMSBoxWithClass())
    # self.add(pr.FilterBoxes(class_names, score_thresh))
    # self.add(pr.ToBoxes2D(class_names, box_method))

    detections = jp.squeeze(detections, axis=0)
    detections = paz.detection.decode(detections, model.prior_boxes, variances)
    detections = remove_class(detections, class_arg)


def DetectSingleShot(
    model,
    class_names,
    score_thresh,
    nms_thresh,
    preprocess=None,
    postprocess=None,
    variances=[0.1, 0.1, 0.2, 0.2],
    draw=True,
):
    if preprocess is None:
        preprocess = paz.lock(SSDPreprocess, model.input_shape[1:3])
    if postprocess is None:
        postprocess = SSDPostprocess(
            model, class_names, score_thresh, nms_thresh
        )

    predict = pr.Predict(self.model, preprocess, postprocess)
    denormalize = pr.DenormalizeBoxes2D()
    draw_boxes2D = pr.DrawBoxes2D(self.class_names)
    wrap = pr.WrapOutput(["image", "boxes2D"])

    def call_with_draw(image):
        boxes2D = predict(image)
        boxes2D = denormalize(image, boxes2D)
        if draw:
            image = draw_boxes2D(image, boxes2D)
        return wrap(image, boxes2D)

    def call_without_draw(image):
        boxes2D = predict(image)
        boxes2D = denormalize(image, boxes2D)
        return wrap(image, boxes2D)

    return call_with_draw if draw else call_without_draw
