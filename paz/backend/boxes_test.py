import importlib
import jax.numpy as jp
import pytest
from paz.backend.boxes import (
    apply_non_max_suppression,
    nms_per_class,
    to_center_form,
    to_corner_form,
    encode,
    decode,
)


# Test apply_non_max_suppression
def test_apply_non_max_suppression():
    boxes = jp.array(
        [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]
    )
    scores = jp.array([0.8, 0.9, 0.7])
    selected_indices, count = apply_non_max_suppression(
        boxes, scores, 0.5, 200
    )
    assert count == 2
    assert selected_indices[0] == 1 and selected_indices[1] == 2


def test_apply_non_max_suppression_empty():
    boxes = jp.array([]).reshape(0, 4)
    scores = jp.array([])
    indices, count = apply_non_max_suppression(boxes, scores)
    assert count == 0


# Test nms_per_class
def test_nms_per_class():
    box_data = jp.array(
        [
            [0.0, 0.0, 1.0, 1.0, 0.1, 0.9],
            [0.5, 0.5, 1.5, 1.5, 0.8, 0.2],
            [2.0, 2.0, 3.0, 3.0, 0.7, 0.3],
        ]
    )
    output = nms_per_class(box_data, 0.5, 0.01, 200)
    assert output.shape == (2, 200, 5)
    # Check if class 1 (index 1) has the high-scoring box


# Test to_center_form and to_corner_form
def test_to_center_form():
    boxes = jp.array([[0.0, 0.0, 2.0, 2.0]])
    center_boxes = to_center_form(boxes)
    expected = jp.array([[1.0, 1.0, 2.0, 2.0]])
    assert jp.allclose(center_boxes, expected)


def test_to_corner_form():
    boxes = jp.array([[1.0, 1.0, 2.0, 2.0]])
    corner_boxes = to_corner_form(boxes)
    expected = jp.array([[0.0, 0.0, 2.0, 2.0]])
    assert jp.allclose(corner_boxes, expected)


# Test encode and decode functions
def test_encode_decode():
    matched = jp.array([[0.0, 0.0, 2.0, 2.0, 1.0]])
    priors = to_center_form(matched[:, :4])
    encoded = encode(matched, priors)
    decoded = decode(encoded, priors)
    assert jp.allclose(decoded[:, :4], matched[:, :4], atol=1e-4)


# Test compute_ious
@pytest.mark.skipif(
    not importlib.util.find_spec("compute_ious"),
    reason="requires the compute_ious",
)
def test_compute_ious():
    boxes_A = jp.array([[0.0, 0.0, 2.0, 2.0]])
    boxes_B = jp.array([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])
    ious = compute_ious(boxes_A, boxes_B)
    expected = jp.array([[1.0, 1 / 7]])
    assert jp.allclose(ious, expected, atol=1e-4)


# Test compute_max_matches
@pytest.mark.skipif(
    not importlib.util.find_spec("compute_max_matches"),
    reason="requires the compute_max_matches",
)
def test_compute_max_matches():
    boxes = jp.array([[0.0, 0.0, 2.0, 2.0]])
    prior_boxes = jp.array([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])
    iou, arg = compute_max_matches(boxes, prior_boxes)
    assert iou.shape == (2,)
    assert arg.shape == (2,)
    assert jp.allclose(iou, jp.array([1.0, 1 / 7]), atol=1e-4)


# Test get_matches_masks
@pytest.mark.skipif(
    not importlib.util.find_spec("get_matches_masks"),
    reason="requires the get_matches_masks",
)
def test_get_matches_masks():
    boxes = jp.array(
        [[0.0, 0.0, 2.0, 2.0, 1.0]]
    )  # Ground truth in corner form
    # Prior boxes in CENTER FORM (x_center, y_center, width, height)
    prior_boxes = jp.array(
        [
            [1.0, 1.0, 2.0, 2.0],  # Converts to [0, 0, 2, 2] in corner form
            [2.0, 2.0, 2.0, 2.0],
        ]
    )  # Converts to [1, 1, 3, 3] in corner form
    matched_arg, pos_mask, ignore_mask = get_matches_masks(
        boxes, prior_boxes, 0.5, 0.4
    )
    assert pos_mask[0] == True  # IoU is 1.0 (perfect overlap)
    assert pos_mask[1] == False  # IoU is 1/7 ≈ 0.14 (no overlap)
    assert ignore_mask[1] == False  # Negative mask (IoU < 0.4)


# Test mask_classes
@pytest.mark.skipif(
    not importlib.util.find_spec("mask_classes"),
    reason="requires the mask_classes",
)
def test_mask_classes():
    matched_boxes = jp.array(
        [[0.0, 0.0, 2.0, 2.0, 1.0], [1.0, 1.0, 3.0, 3.0, 2.0]]
    )
    positive_mask = jp.array([True, False])
    ignoring_mask = jp.array([False, False])
    masked = mask_classes(matched_boxes, positive_mask, ignoring_mask)
    assert masked[0, 4] == 1.0
    assert masked[1, 4] == 0.0


# Test match and match2 functions
@pytest.mark.skipif(
    not importlib.util.find_spec("match"),
    reason="requires the match",
)
def test_match():
    boxes = jp.array([[0.0, 0.0, 2.0, 2.0, 1.0]])
    prior_boxes = jp.array([[1.0, 1.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]])
    matched = match(boxes, prior_boxes, 0.5, 0.4)
    assert matched.shape == (2, 5)
    assert (
        matched[0, 4] == 1.0
    )  # IoU is (1/7) which is ~0.142 <0.5, so negative


@pytest.mark.skipif(
    not importlib.util.find_spec("match2"),
    reason="requires the match2",
)
def test_match2():
    boxes = jp.array([[0.0, 0.0, 2.0, 2.0, 1.0], [1.0, 1.0, 3.0, 3.0, 2.0]])
    prior_boxes = jp.array([[1.0, 1.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]])
    matched = match2(boxes, prior_boxes, 0.5)
    assert matched.shape == (2, 5)
    # Check if class is set to 0 where IoU < threshold


# Test compute_iou
@pytest.mark.skipif(
    not importlib.util.find_spec("compute_iou"),
    reason="requires the compute_iou",
)
def test_compute_iou():
    box = jp.array([0.0, 0.0, 2.0, 2.0])
    boxes = jp.array([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])
    ious = compute_iou(box, boxes)
    assert jp.allclose(ious, jp.array([1.0, 1 / 7]), atol=1e-4)


# Test to_one_hot
@pytest.mark.skipif(
    not importlib.util.find_spec("to_one_hot"),
    reason="requires the to_one_hot",
)
def test_to_one_hot():
    class_indices = jp.array([1, 3])
    one_hot = to_one_hot(class_indices, 4)
    expected = jp.array([[0, 1, 0, 0], [0, 0, 0, 1]])
    assert jp.allclose(one_hot, expected)


# Test make_box_square
@pytest.mark.skipif(
    not importlib.util.find_spec("make_box_square"),
    reason="requires the make_box_square",
)
def test_make_box_square():
    box = (0, 0, 4, 2)
    square_box = make_box_square(jp.array(box))
    # Expected to adjust y coordinates to match width
    expected = (0, -1, 4, 3)
    assert jp.array_equal(square_box, jp.array(expected))


# Test offset
@pytest.mark.skipif(
    not importlib.util.find_spec("offset"),
    reason="requires the offset",
)
def test_offset():
    coords = (10, 20, 30, 40)
    offset_scales = (0.1, 0.1)
    new_coords = offset(coords, offset_scales)
    # x offset is (30-10)*0.1 = 2, y offset is (40-20)*0.1 = 2
    # x_min -2, x_max +2; y_min -2, y_max +2
    expected = (8, 18, 32, 42)
    assert jp.array_equal(new_coords, jp.array(expected))


# Test clip
@pytest.mark.skipif(
    not importlib.util.find_spec("clip"),
    reason="requires the clip",
)
def test_clip():
    coords = (-10, -5, 150, 200)
    image_shape = (100, 100)
    clipped = clip(coords, image_shape)
    expected = (0, 0, 100, 100)
    assert jp.array_equal(clipped, jp.array(expected))


# Test denormalize_box
@pytest.mark.skipif(
    not importlib.util.find_spec("denormalize_box"),
    reason="requires the denormalize_box",
)
def test_denormalize_box():
    box = jp.array([0.5, 0.5, 1.0, 1.0])
    image_shape = (100, 200)
    denorm_box = denormalize_box(box, image_shape)
    expected = (100, 50, 200, 100)
    assert jp.array_equal(denorm_box, jp.array(expected))


# Test flip_left_right
@pytest.mark.skipif(
    not importlib.util.find_spec("flip_left_right"),
    reason="requires the flip_left_right",
)
def test_flip_left_right():
    boxes = jp.array([[10.0, 20.0, 30.0, 40.0]])
    width = 100
    flipped = flip_left_right(boxes, width)
    expected = jp.array([[70.0, 20.0, 90.0, 40.0]])
    assert jp.allclose(flipped, expected)


# Test to_image_coordinates and to_normalized_coordinates
@pytest.mark.skipif(
    not all(
        importlib.util.find_spec(func)
        for func in ["to_image_coordinates", "to_normalized_coordinates"]
    ),
    reason="requires both to_image_coordinates and to_normalized_coordinates",
)
def test_coordinate_conversions():
    image = jp.zeros((100, 200, 3))
    boxes = jp.array([[0.5, 0.5, 1.0, 1.0]])
    image_boxes = to_image_coordinates(boxes, image)
    normalized_boxes = to_normalized_coordinates(image_boxes, image)
    assert jp.allclose(normalized_boxes, boxes, atol=1e-4)
