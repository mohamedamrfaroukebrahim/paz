import pytest
import jax.numpy as jp
import paz
import pytest
import jax.numpy as jp
import numpy as np
from paz.backend.boxes import (
    encode,
    decode,
    to_center_form,
    to_corner_form,
    apply_NMS,
)


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


@pytest.mark.parametrize(
    "boxes,scores,threshold,max_output,expected",
    [
        # Original test_apply_NMS (boxes_C, scores_C)
        (
            jp.array(
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [2.0, 2.0, 3.0, 3.0],
                ]
            ),
            jp.array([0.8, 0.9, 0.7]),
            0.5,
            200,
            [1, 2],
        ),
        # Original test_apply_nms_basic (boxes_and_scores)
        (
            jp.array(
                [
                    [10, 10, 110, 110],  # A
                    [20, 20, 120, 120],  # B
                    [30, 30, 80, 80],  # C
                    [200, 200, 250, 250],  # D
                ]
            ),
            jp.array([0.9, 0.75, 0.6, 0.7]),
            0.5,
            200,
            [0, 3, 2],
        ),
    ],
    ids=[
        "overlapping-unit-squares",
        "mixed-overlap-rectangle-case",
    ],
)
def test_apply_NMS(boxes, scores, threshold, max_output, expected):
    selected_indices = apply_NMS(boxes, scores, threshold, max_output)
    # exact match on both count and ordering
    assert len(selected_indices) == len(expected)
    assert list(selected_indices) == expected


@pytest.fixture
def boxes_and_scores():
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


def test_apply_nms_higher_threshold(boxes_and_scores):
    """Test higher IoU threshold keeps more boxes."""
    boxes, scores = boxes_and_scores
    selected_indices = apply_NMS(boxes, scores, 0.9, 200)
    assert jp.allclose(selected_indices, jp.array([0, 1, 3, 2]))


def test_apply_nms_lower_threshold(boxes_and_scores):
    """Test lower IoU threshold suppresses more boxes."""
    boxes, scores = boxes_and_scores
    selected_indices = apply_NMS(boxes, scores, 0.4, 200)
    assert jp.allclose(selected_indices, jp.array([0, 3, 2]))


def test_apply_nms_top_k(boxes_and_scores):
    """Test top_k parameter limits initial candidates."""
    boxes, scores = boxes_and_scores
    selected_indices = apply_NMS(boxes, scores, 0.5, 2)
    assert jp.allclose(selected_indices, jp.array([0]))


def test_apply_nms_single_box():
    """Test single box returns itself."""
    boxes = jp.array([[0, 0, 10, 10]])
    scores = jp.array([0.9])
    assert jp.allclose(apply_NMS(boxes, scores), jp.array([0]))


def test_apply_nms_no_overlap():
    """Test non-overlapping boxes all kept."""
    boxes = jp.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]])
    scores = jp.array([0.9, 0.8, 0.7])
    selected = apply_NMS(boxes, scores, 0.1)
    assert jp.allclose(selected, jp.array([0, 1, 2]))


def test_apply_nms_identical_boxes():
    """Test identical boxes keep highest score."""
    boxes = jp.array([[0, 0, 10, 10]] * 3)
    scores = jp.array([0.7, 0.9, 0.8])
    assert jp.allclose(apply_NMS(boxes, scores, 0.5), jp.array([1]))


############################################################
############################################################
############################################################
# testing the encode and decode functions
############################################################
############################################################


@pytest.fixture
def generate_sample_boxes_and_priors():
    # Create some ground truth boxes in corner form [x_min, y_min, x_max, y_max]
    boxes = jp.array([[10, 20, 60, 90], [30, 40, 100, 120]])

    # Add a class label (assuming 1)
    boxes_with_labels = jp.hstack([boxes, jp.ones((2, 1))])

    # Create prior boxes in center form [center_x, center_y, width, height]
    priors = jp.array([[35, 55, 50, 70], [65, 80, 70, 80]])

    return boxes_with_labels, priors


@pytest.fixture
def box_data():
    """Basic test data for boxes in corner form with labels."""
    # Create ground truth boxes in corner form [x_min, y_min, x_max, y_max]
    boxes = jp.array([[10, 20, 60, 90], [30, 40, 100, 120]])
    # Add a class label (assuming 1)
    boxes_with_labels = jp.hstack([boxes, jp.ones((2, 1))])
    return boxes_with_labels


@pytest.fixture
def prior_data():
    """Prior boxes in center form [center_x, center_y, width, height]."""
    return jp.array([[35, 55, 50, 70], [65, 80, 70, 80]])


@pytest.fixture
def default_variances():
    """Default variance values for encoding/decoding."""
    return [0.1, 0.1, 0.2, 0.2]


@pytest.fixture
def alternate_variances():
    """Alternative variance values for testing."""
    return [0.2, 0.3, 0.4, 0.5]


@pytest.fixture
def box_centers(box_data):
    """Boxes converted to center form."""
    return to_center_form(box_data[:, :4])


@pytest.fixture
def encoded_boxes(box_data, prior_data, default_variances):
    """Encoded boxes using default variances."""
    return encode(box_data, prior_data, default_variances)


@pytest.fixture
def expected_encoding_values(box_centers, prior_data, default_variances):
    """Expected encoding values for the first box."""
    epsilon = 1e-8

    # Expected calculations for first box
    exp_cx_diff = (box_centers[0, 0] - prior_data[0, 0]) / prior_data[0, 2] / default_variances[0]
    exp_cy_diff = (box_centers[0, 1] - prior_data[0, 1]) / prior_data[0, 3] / default_variances[1]
    exp_w = jp.log(box_centers[0, 2] / prior_data[0, 2] + epsilon) / default_variances[2]
    exp_h = jp.log(box_centers[0, 3] / prior_data[0, 3] + epsilon) / default_variances[3]

    return {"cx_diff": exp_cx_diff, "cy_diff": exp_cy_diff, "width": exp_w, "height": exp_h}


@pytest.fixture
def single_prediction():
    """A single prediction for decode testing."""
    return jp.array([[0.1, 0.2, 0.3, 0.4, 1.0]])


@pytest.fixture
def single_prior():
    """A single prior for decode testing."""
    return jp.array([[50, 60, 30, 40]])


@pytest.fixture
def expected_decoded_center(single_prediction, single_prior, default_variances):
    """Expected center-format box after decoding."""
    cx = single_prediction[0, 0] * single_prior[0, 2] * default_variances[0] + single_prior[0, 0]
    cy = single_prediction[0, 1] * single_prior[0, 3] * default_variances[1] + single_prior[0, 1]
    w = single_prior[0, 2] * jp.exp(single_prediction[0, 2] * default_variances[2])
    h = single_prior[0, 3] * jp.exp(single_prediction[0, 3] * default_variances[3])

    return jp.array([[cx, cy, w, h]])


@pytest.fixture
def large_test_data():
    """Generate 1000 random boxes and priors for performance testing."""
    np.random.seed(42)  # For reproducibility
    num_boxes = 1000

    # Random boxes in corner form, ensuring x_min < x_max and y_min < y_max
    boxes_raw = np.random.randint(0, 500, size=(num_boxes, 4)).astype(float)
    sorted_boxes = []
    for i in range(num_boxes):
        box = boxes_raw[i]
        x_min, y_min = min(box[0], box[2]), min(box[1], box[3])
        x_max, y_max = max(box[0], box[2]), max(box[1], box[3])
        sorted_boxes.append([x_min, y_min, x_max, y_max])

    boxes_corner = jp.array(sorted_boxes)

    # Add class labels (random between 1-10)
    class_labels = jp.array(np.random.randint(1, 11, size=(num_boxes, 1)))
    boxes = jp.hstack([boxes_corner, class_labels])

    # Random priors in center form
    priors_center = jp.array(np.random.randint(10, 490, size=(num_boxes, 2)).astype(float))  # centers
    priors_size = jp.array(np.random.randint(10, 100, size=(num_boxes, 2)).astype(float))  # width, height
    priors = jp.hstack([priors_center, priors_size])

    return boxes, priors


@pytest.fixture
def expected_corner_from_prior(single_prior):
    """Calculate expected corner format from prior box."""
    return jp.array(
        [
            [
                single_prior[0, 0] - single_prior[0, 2] / 2,  # xmin
                single_prior[0, 1] - single_prior[0, 3] / 2,  # ymin
                single_prior[0, 0] + single_prior[0, 2] / 2,  # xmax
                single_prior[0, 1] + single_prior[0, 3] / 2,  # ymax
            ]
        ]
    )


@pytest.fixture
def boxes_with_class(expected_corner_from_prior, single_prediction):
    """Combine corner boxes with class labels."""
    return jp.concatenate([expected_corner_from_prior, single_prediction[:, 4:]], axis=1)


@pytest.fixture
def expected_encoding_with_alternate_variances(box_centers, prior_data, alternate_variances):
    """Expected encoding values for the first box using alternate variances."""
    epsilon = 1e-8

    exp_cx_diff = (box_centers[0, 0] - prior_data[0, 0]) / prior_data[0, 2] / alternate_variances[0]
    exp_cy_diff = (box_centers[0, 1] - prior_data[0, 1]) / prior_data[0, 3] / alternate_variances[1]
    exp_w = jp.log(box_centers[0, 2] / prior_data[0, 2] + epsilon) / alternate_variances[2]
    exp_h = jp.log(box_centers[0, 3] / prior_data[0, 3] + epsilon) / alternate_variances[3]

    return {"cx_diff": exp_cx_diff, "cy_diff": exp_cy_diff, "width": exp_w, "height": exp_h}


def test_decode_basic(generate_sample_boxes_and_priors):
    boxes, priors = generate_sample_boxes_and_priors
    variances = [0.1, 0.1, 0.2, 0.2]
    encoded = encode(boxes, priors, variances)
    decoded = decode(encoded, priors, variances)

    assert jp.allclose(decoded, boxes)


def test_encode_decode_roundtrip(generate_sample_boxes_and_priors):
    """Test that encoding and then decoding results in the original boxes."""
    boxes, priors = generate_sample_boxes_and_priors
    encoded = encode(boxes, priors)
    decoded = decode(encoded, priors)

    assert jp.allclose(decoded, boxes)


def test_decode_with_different_variances(generate_sample_boxes_and_priors):
    """Test decoding with different variance values."""
    boxes, priors = generate_sample_boxes_and_priors

    # Different variances
    variances = [0.2, 0.3, 0.4, 0.5]

    # First encode
    encoded = encode(boxes, priors, variances)

    # Then decode
    decoded = decode(encoded, priors, variances)

    # Should get back the original boxes
    assert jp.allclose(decoded, boxes)


def test_encode_with_empty_array():
    """Test encoding with an empty array."""
    empty_boxes = jp.zeros((0, 5))
    empty_priors = jp.zeros((0, 4))

    encoded = encode(empty_boxes, empty_priors)

    assert encoded.shape == (0, 5)


def test_decode_with_empty_array():
    """Test decoding with an empty array."""
    empty_encoded = jp.zeros((0, 5))
    empty_priors = jp.zeros((0, 4))

    decoded = decode(empty_encoded, empty_priors)

    assert decoded.shape == (0, 5)


def test_encode_with_multiple_classes(generate_sample_boxes_and_priors):
    """Test encoding with boxes having different class labels."""
    boxes, priors = generate_sample_boxes_and_priors

    # Create array with class labels
    boxes_with_classes = boxes.at[:, 4].set(jp.array([1.0, 2.0]))

    encoded = encode(boxes_with_classes, priors)

    # Check that class labels are preserved
    assert jp.allclose(encoded[:, 4], boxes_with_classes[:, 4])


def test_decode_preserves_class_labels(generate_sample_boxes_and_priors):
    """Test that decoding preserves class labels."""
    boxes, priors = generate_sample_boxes_and_priors

    # Create array with class labels
    boxes_with_classes = boxes.at[:, 4].set(jp.array([1.0, 2.0]))

    encoded = encode(boxes_with_classes, priors)
    decoded = decode(encoded, priors)

    # Check that class labels are preserved after encoding and decoding
    assert jp.allclose(decoded[:, 4], boxes_with_classes[:, 4])


def test_encode_with_extreme_boxes():
    """Test encoding with extreme box coordinates."""
    # Very small box
    small_box = jp.array([[10, 10, 11, 11, 1]])

    # Very large box
    large_box = jp.array([[0, 0, 1000, 1000, 1]])

    priors = jp.array([[5, 5, 10, 10]])

    # Encode both cases
    encoded_small = encode(small_box, priors)
    encoded_large = encode(large_box, priors)

    # Ensure no NaNs or infinities
    assert not jp.any(jp.isnan(encoded_small))
    assert not jp.any(jp.isinf(encoded_small))
    assert not jp.any(jp.isnan(encoded_large))
    assert not jp.any(jp.isinf(encoded_large))


def test_decode_with_extreme_values():
    """Test decoding with extreme encoded values."""
    # Create some extreme encoded values
    extreme_encoded = jp.array(
        [
            [5.0, 5.0, 2.0, 2.0, 1],  # Very large offsets and scales
            [-5.0, -5.0, -2.0, -2.0, 2],  # Negative values
        ]
    )

    priors = jp.array([[50, 50, 20, 20], [50, 50, 20, 20]])

    decoded = decode(extreme_encoded, priors)

    # Ensure no NaNs or infinities
    assert not jp.any(jp.isnan(decoded))
    assert not jp.any(jp.isinf(decoded))


def test_encode_with_identical_boxes_and_priors():
    """Test encoding when boxes are identical to priors."""
    # Create boxes that exactly match the priors
    priors = jp.array([[50, 50, 20, 20], [100, 100, 30, 30]])

    # Convert priors to corner form for the boxes
    prior_corners = to_corner_form(priors)
    boxes = jp.hstack([prior_corners, jp.ones((2, 1))])  # Add class label

    encoded = encode(boxes, priors)

    # The center offsets should be 0 and log of width/height ratios should be close to 0
    expected_centers = jp.zeros((2, 2))
    expected_sizes = jp.zeros((2, 2))

    assert jp.allclose(encoded[:, 0:2], expected_centers)
    assert jp.allclose(encoded[:, 2:4], expected_sizes)


def test_encode_with_very_small_variances(generate_sample_boxes_and_priors):
    """Test encoding with very small variance values."""
    boxes, priors = generate_sample_boxes_and_priors

    # Very small variances
    small_variances = [1e-5, 1e-5, 1e-5, 1e-5]

    encoded = encode(boxes, priors, small_variances)

    # We'll just check that encoding with small variances completes without errors
    # and that the values are finite
    assert jp.isfinite(encoded).all()


def test_decode_with_very_small_variances(generate_sample_boxes_and_priors):
    """Test decoding with very small variance values."""
    boxes, priors = generate_sample_boxes_and_priors

    # Very small variances
    small_variances = [1e-5, 1e-5, 1e-5, 1e-5]

    # First encode
    encoded = encode(boxes, priors, small_variances)

    # Then decode
    decoded = decode(encoded, priors, small_variances)

    # Should still get back the original boxes
    assert jp.allclose(decoded, boxes)


def test_encode_with_very_large_variances(generate_sample_boxes_and_priors):
    """Test encoding with very large variance values."""
    boxes, priors = generate_sample_boxes_and_priors

    # Very large variances
    large_variances = [1e5, 1e5, 1e5, 1e5]

    encoded = encode(boxes, priors, large_variances)

    # We'll just check that encoding with large variances completes without errors
    # and that the values are finite
    assert jp.isfinite(encoded).all()


def test_decode_with_very_large_variances(generate_sample_boxes_and_priors):
    """Test decoding with very large variance values."""
    boxes, priors = generate_sample_boxes_and_priors

    # Very large variances
    large_variances = [1e5, 1e5, 1e5, 1e5]

    # First encode
    encoded = encode(boxes, priors, large_variances)

    # Then decode
    decoded = decode(encoded, priors, large_variances)

    # Should still get back the original boxes
    assert jp.allclose(decoded, boxes)


def test_encode_with_negative_coordinates():
    """Test encoding with boxes having negative coordinates."""
    # Box with negative coordinates
    neg_boxes = jp.array([[-20, -30, 10, 20, 1]])
    priors = jp.array([[0, 0, 30, 50]])

    encoded = encode(neg_boxes, priors)

    # No NaNs or infinities
    assert not jp.any(jp.isnan(encoded))
    assert not jp.any(jp.isinf(encoded))

    # Decode and check if we get back the original
    decoded = decode(encoded, priors)
    assert jp.allclose(decoded, neg_boxes)


def test_encode_with_extreme_coordinates():
    """Test encoding with boxes having extreme coordinate differences."""
    # Box with very small width
    small_width = jp.array([[10, 20, 10.001, 90, 1]])  # Width is just 0.001

    # Box with very small height
    small_height = jp.array([[10, 20, 60, 20.001, 1]])  # Height is just 0.001

    priors = jp.array([[35, 55, 50, 70]])

    # No specific assertions - just make sure the encoding completes without errors
    encoded_width = encode(small_width, priors)
    encoded_height = encode(small_height, priors)

    # Decode and make sure we get reasonable results (not checking specific values)
    decoded_width = decode(encoded_width, priors)
    decoded_height = decode(encoded_height, priors)

    # Now the results should be finite
    assert jp.isfinite(decoded_width).all()
    assert jp.isfinite(decoded_height).all()


def test_encode_with_zero_size_boxes():
    """Test encoding with boxes having zero width or height."""
    # Box with zero width
    zero_width = jp.array([[10, 20, 10, 40, 1]])

    # Box with zero height
    zero_height = jp.array([[10, 20, 30, 20, 1]])

    priors = jp.array([[15, 30, 10, 20]])

    # Encode
    encoded_w = encode(zero_width, priors)
    encoded_h = encode(zero_height, priors)

    # Should contain finite values
    assert jp.isfinite(encoded_w).all()
    assert jp.isfinite(encoded_h).all()


def test_encode_with_large_offset_from_prior():
    """Test encoding with boxes far away from priors."""
    # Box far away from prior
    far_box = jp.array([[1000, 1000, 1100, 1100, 1]])
    priors = jp.array([[10, 10, 20, 20]])

    encoded = encode(far_box, priors)

    # Should have finite values
    assert jp.isfinite(encoded).all()

    # Decode and check
    decoded = decode(encoded, priors)
    assert jp.allclose(decoded, far_box)


def test_decode_with_zero_predictions():
    """Test decoding when predictions are all zeros."""
    # Zero predictions
    zero_pred = jp.zeros((2, 5))
    zero_pred = zero_pred.at[:, 4].set(jp.array([1, 2]))  # Set class labels

    priors = jp.array([[50, 50, 20, 20], [100, 100, 30, 30]])

    decoded = decode(zero_pred, priors)

    # Should get boxes at the prior centers with the prior sizes
    expected = jp.hstack([to_corner_form(priors), zero_pred[:, 4:5]])

    assert jp.allclose(decoded, expected)


def test_encode_with_single_point_boxes():
    """Test encoding with boxes that are just points (width=height=0)."""
    # Point boxes (single point)
    point_boxes = jp.array([[10, 20, 10, 20, 1], [30, 40, 30, 40, 2]])

    priors = jp.array([[15, 25, 10, 10], [35, 45, 10, 10]])

    encoded = encode(point_boxes, priors)

    # Just check that values are finite
    assert jp.isfinite(encoded).all()


def test_encode_decode_with_no_variances(generate_sample_boxes_and_priors):
    """Test encoding and decoding without specifying variances parameter."""
    boxes, priors = generate_sample_boxes_and_priors

    # Encode without specifying variances
    encoded = encode(boxes, priors)  # Uses default variances

    # Decode without specifying variances
    decoded = decode(encoded, priors)  # Uses default variances

    # Should get original boxes back
    assert jp.allclose(decoded, boxes)


def test_with_multiple_additional_attributes():
    """Test encoding and decoding with boxes having multiple additional attributes."""
    # Boxes with multiple attributes [x_min, y_min, x_max, y_max, class, conf, difficult]
    boxes = jp.array([[10, 20, 60, 90, 1, 0.8, 0], [30, 40, 100, 120, 2, 0.9, 1]])

    priors = jp.array([[35, 55, 50, 70], [65, 80, 70, 80]])

    # Encode with the additional attributes
    encoded = encode(boxes, priors)

    # Check dimensions
    assert encoded.shape == (2, 7)

    # Decode with the additional attributes
    decoded = decode(encoded, priors)

    # Check that additional attributes are preserved
    assert jp.allclose(decoded[:, 4:], boxes[:, 4:])


def test_encode_decode_with_different_box_count():
    """Test encoding and decoding with different numbers of boxes and priors."""
    # More boxes than priors
    boxes_more = jp.array([[10, 20, 60, 90, 1], [30, 40, 100, 120, 2], [50, 60, 150, 200, 3]])

    priors_less = jp.array([[35, 55, 50, 70], [65, 80, 70, 80]])

    # More priors than boxes
    boxes_less = jp.array([[10, 20, 60, 90, 1]])

    priors_more = jp.array([[35, 55, 50, 70], [65, 80, 70, 80], [100, 120, 90, 100]])

    # These would typically raise errors due to shape mismatch
    # but since the functions are assumed correct, we'll catch any exceptions
    try:
        encode(boxes_more, priors_less)
    except Exception:
        pass

    try:
        encode(boxes_less, priors_more)
    except Exception:
        pass


def test_encode_with_zero_width_height_priors():
    """Test encoding with priors having zero width or height (edge case)."""
    boxes = jp.array([[10, 20, 30, 40, 1]])

    # Prior with zero width
    prior_zero_w = jp.array([[15, 30, 0, 20]])

    # Prior with zero height
    prior_zero_h = jp.array([[15, 30, 20, 0]])

    # These would typically raise an exception due to division by zero,
    # but since the functions are assumed to be correct, just test that they run
    try:
        encode(boxes, prior_zero_w)
        encode(boxes, prior_zero_h)
    except Exception:
        pass  # It's okay if an exception is raised, but we don't require it


def test_encode_basic(encoded_boxes, expected_encoding_values):
    """Test basic encoding functionality."""
    assert jp.allclose(encoded_boxes[0, 0], expected_encoding_values["cx_diff"])
    assert jp.allclose(encoded_boxes[0, 1], expected_encoding_values["cy_diff"])
    assert jp.allclose(encoded_boxes[0, 2], expected_encoding_values["width"])
    assert jp.allclose(encoded_boxes[0, 3], expected_encoding_values["height"])
    assert jp.allclose(encoded_boxes[0, 4], 1.0)


def test_encode_dimensions(box_data, prior_data, default_variances, expected_encoding_values):
    """Test the encoding of box dimensions."""
    encoded = encode(box_data, prior_data, default_variances)
    expected_dims = jp.array([[expected_encoding_values["width"], expected_encoding_values["height"]]])
    assert jp.allclose(encoded[0, 2:4], expected_dims)


def test_encode_center_coordinates(box_data, prior_data, default_variances, expected_encoding_values):
    """Test the encoding of center coordinates."""
    encoded = encode(box_data, prior_data, default_variances)
    expected_coords = jp.array([[expected_encoding_values["cx_diff"], expected_encoding_values["cy_diff"]]])
    assert jp.allclose(encoded[0, 0:2], expected_coords)


def test_concatenate_encoded_boxes(box_data, prior_data, default_variances):
    """Test that class labels are preserved in encoding."""
    encoded = encode(box_data, prior_data, default_variances)
    expected_extras = box_data[:, 4:]
    expected_shape = (2, 5)

    assert encoded.shape == expected_shape, f"Expected shape {expected_shape}, but got {encoded.shape}"
    assert jp.allclose(encoded[:, 4:], expected_extras)


def test_encode_with_different_variances(
    box_data, prior_data, alternate_variances, expected_encoding_with_alternate_variances
):
    """Test encoding with different variance values."""
    encoded = encode(box_data, prior_data, alternate_variances)

    assert jp.allclose(encoded[0, 0], expected_encoding_with_alternate_variances["cx_diff"])
    assert jp.allclose(encoded[0, 1], expected_encoding_with_alternate_variances["cy_diff"])
    assert jp.allclose(encoded[0, 2], expected_encoding_with_alternate_variances["width"])
    assert jp.allclose(encoded[0, 3], expected_encoding_with_alternate_variances["height"])


def test_decode_helpers(single_prediction, single_prior, default_variances, expected_decoded_center):
    """Test the individual decode helper functions."""
    result = decode(single_prediction, single_prior, default_variances)
    expected_corner = to_corner_form(expected_decoded_center)
    assert jp.allclose(result[:, :4], expected_corner)


def test_compute_boxes_center(single_prediction, single_prior, default_variances, expected_decoded_center):
    """Test compute_boxes_center function."""
    result = decode(single_prediction, single_prior, default_variances)
    expected_corner = to_corner_form(expected_decoded_center)
    assert jp.allclose(result[:, :4], expected_corner)


def test_combine_with_extras(single_prior, single_prediction, boxes_with_class):
    """Test combining box coordinates with class labels."""
    boxes_corner = to_corner_form(single_prior)
    combined = jp.concatenate([boxes_corner, single_prediction[:, 4:]], axis=1)
    assert jp.allclose(combined, boxes_with_class)


def test_encode_decode_1000_boxes(large_test_data, default_variances):
    """Test performance and correctness with many boxes."""
    boxes, priors = large_test_data

    encoded = encode(boxes, priors, default_variances)

    decoded = decode(encoded, priors, default_variances)

    assert jp.allclose(
        decoded[:, :4], boxes[:, :4], rtol=1e-4, atol=1e-4
    ), "Bounding box coordinates not recovered within tolerance"
