import pytest
import jax.numpy as jp
import paz


@pytest.fixture
def boxes_A():
    return jp.array(
        [
            [54, 66, 198, 114],
            [42, 78, 186, 126],
            [18, 63, 235, 135],
            [18, 63, 235, 135],
            [54, 72, 198, 120],
            [36, 60, 180, 108],
        ]
    )


@pytest.fixture
def boxes_B():
    return jp.array(
        [
            [39, 63, 203, 112],
            [49, 75, 203, 125],
            [31, 69, 201, 125],
            [50, 72, 197, 121],
            [35, 51, 196, 110],
        ]
    )


@pytest.fixture
def true_IOUs():
    return jp.array([0.48706725, 0.787838, 0.70033113, 0.70739083, 0.39040922])


@pytest.fixture
def boxes_D_corner_form():
    return jp.array([[0.0, 0.0, 2.0, 2.0]])


@pytest.fixture
def boxes_D_center_form():
    return jp.array([[1.0, 1.0, 2.0, 2.0]])


def test_compute_IOUs_self_intersection_A(boxes_A):
    pred_IOUs = paz.boxes.compute_IOUs(boxes_A, boxes_A)
    assert jp.allclose(1.0, jp.diag(pred_IOUs))


def test_compute_IOUs_self_intersection_B(boxes_B):
    pred_IOUs = paz.boxes.compute_IOUs(boxes_B, boxes_B)
    assert jp.allclose(1.0, jp.diag(pred_IOUs))


def test_compute_IOUs_shape(boxes_A, boxes_B):
    pred_IOUs = paz.boxes.compute_IOUs(boxes_A, boxes_B)
    assert len(pred_IOUs.shape) == 2
    num_rows, num_cols = pred_IOUs.shape
    assert num_rows == len(boxes_A)
    assert num_cols == len(boxes_B)


def test_compute_ious(boxes_A, boxes_B, true_IOUs):
    pred_IOUs = paz.boxes.compute_IOUs(boxes_A[1:2, :], boxes_B)
    assert jp.allclose(true_IOUs, pred_IOUs)


def test_to_center_form(boxes_D_corner_form, boxes_D_center_form):
    values = paz.boxes.to_center_form(boxes_D_corner_form)
    assert jp.allclose(boxes_D_center_form, values)


def test_to_corner_form(boxes_D_corner_form, boxes_D_center_form):
    values = paz.boxes.to_corner_form(boxes_D_center_form)
    assert jp.allclose(boxes_D_corner_form, values)


@pytest.mark.skipif(
    not importlib.util.find_spec("to_one_hot"),
    reason="requires the to_one_hot",
)
def test_to_one_hot():
    class_indices = jp.array([1, 3])
    one_hot = to_one_hot(class_indices, 4)
    expected = jp.array([[0, 1, 0, 0], [0, 0, 0, 1]])
    assert jp.allclose(one_hot, expected)


@pytest.mark.skipif(
    not importlib.util.find_spec("make_box_square"),
    reason="requires the make_box_square",
)
def test_make_box_square():
    box = (0, 0, 4, 2)
    square_box = paz.boxes.square(jp.array(box))
    # Expected to adjust y coordinates to match width
    expected = (0, -1, 4, 3)
    assert jp.array_equal(square_box, jp.array(expected))


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
