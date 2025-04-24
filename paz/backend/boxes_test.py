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
def boxes_C():
    return jp.array(
        [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]
    )


@pytest.fixture
def scores_C():
    return jp.array([0.8, 0.9, 0.7])


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


def test_apply_NMS(boxes_C, scores_C):
    selected_indices = paz.boxes.apply_NMS(boxes_C, scores_C, 0.5, 200)
    assert len(selected_indices) == 2
    assert selected_indices[0] == 1
    assert selected_indices[1] == 2


def test_to_center_form(boxes_D_corner_form, boxes_D_center_form):
    values = paz.boxes.to_center_form(boxes_D_corner_form)
    assert jp.allclose(boxes_D_center_form, values)


def test_to_corner_form(boxes_D_corner_form, boxes_D_center_form):
    values = paz.boxes.to_corner_form(boxes_D_center_form)
    assert jp.allclose(boxes_D_corner_form, values)


@pytest.mark.skip(reason="Not implemented")
def test_encode_decode():
    matched = jp.array([[0.0, 0.0, 2.0, 2.0, 1.0]])
    priors = paz.boxes.to_center_form(matched[:, :4])
    encoded = paz.boxes.encode(matched, priors)
    decoded = paz.boxes.decode(encoded, priors)
    assert jp.allclose(decoded[:, :4], matched[:, :4], atol=1e-4)


##############################################################################
##############################################################################
# The following tests are for the apply_NMS function in the paz.backend.boxes module.
##############################################################################
##############################################################################
import jax.numpy as jp
import pytest
from paz.backend.boxes import apply_NMS


@pytest.fixture
def simple_boxes_scores():
    """
    Create a simple test case with 4 boxes:
    - Box A: [10, 10, 110, 110] with score 0.9
    - Box B: [20, 20, 120, 120] with score 0.75 (high overlap with A)
    - Box C: [30, 30, 80, 80] with score 0.6 (partial overlap with A and B)
    - Box D: [200, 200, 250, 250] with score 0.7 (no overlap with others)
    """
    boxes = jp.array(
        [
            [10, 10, 110, 110],  # Box A (Area 10000)
            [20, 20, 120, 120],  # Box B (Area 10000), IoU with A ~ 0.81
            [30, 30, 80, 80],  # Box C (Area 2500), IoU with A ~ 0.23, B ~ 0.44
            [200, 200, 250, 250],  # Box D (Area 2500), No overlap
        ]
    )
    scores = jp.array([0.9, 0.75, 0.6, 0.7])
    return boxes, scores


def test_apply_nms_basic(simple_boxes_scores):
    """Test standard NMS scenario."""
    boxes, scores = simple_boxes_scores
    selected_indices = apply_NMS(boxes, scores, 0.5, 200)
    assert set(selected_indices.tolist()) == {0, 3, 2}


def test_apply_nms_higher_threshold(simple_boxes_scores):
    """Test higher IoU threshold keeps more boxes."""
    boxes, scores = simple_boxes_scores
    selected_indices = apply_NMS(boxes, scores, 0.9, 200)
    assert set(selected_indices.tolist()) == {0, 1, 3, 2}


def test_apply_nms_lower_threshold(simple_boxes_scores):
    """Test lower IoU threshold suppresses more boxes."""
    boxes, scores = simple_boxes_scores
    selected_indices = apply_NMS(boxes, scores, 0.4, 200)
    assert set(selected_indices.tolist()) == {0, 3, 2}


def test_apply_nms_top_k(simple_boxes_scores):
    """Test top_k parameter limits initial candidates."""
    boxes, scores = simple_boxes_scores
    selected_indices = apply_NMS(boxes, scores, 0.5, 2)
    assert set(selected_indices.tolist()) == {0}


def test_apply_nms_single_box():
    """Test single box returns itself."""
    boxes = jp.array([[0, 0, 10, 10]])
    scores = jp.array([0.9])
    assert jp.array_equal(apply_NMS(boxes, scores), jp.array([0]))


def test_apply_nms_no_overlap():
    """Test non-overlapping boxes all kept."""
    boxes = jp.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]])
    scores = jp.array([0.9, 0.8, 0.7])
    selected = apply_NMS(boxes, scores, 0.1)
    assert set(selected.tolist()) == {0, 1, 2}


def test_apply_nms_identical_boxes():
    """Test identical boxes keep highest score."""
    boxes = jp.array([[0, 0, 10, 10]] * 3)
    scores = jp.array([0.7, 0.9, 0.8])
    assert jp.array_equal(apply_NMS(boxes, scores, 0.5), jp.array([1]))


def test_apply_nms_zero_area_boxes():
    """Test JAX with zero-area boxes to ensure proper handling of invalid boxes."""
    boxes = jp.array(
        [
            [0, 0, 10, 10],  # Normal
            [5, 5, 5, 15],  # Zero width
            [5, 5, 15, 5],  # Zero height
            [5, 5, 15, 15],  # Normal
        ]
    )
    scores = jp.array([0.9, 0.8, 0.7, 0.6])
    selected = apply_NMS(boxes, scores)
    # Only normal boxes should be selected
    assert 0 in selected and 3 in selected
    # Zero area boxes should be filtered out during the NMS process
    assert 1 not in selected and 2 not in selected

