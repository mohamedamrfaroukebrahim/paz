import jax.numpy as jp
from paz.backend.boxes import (
    apply_non_max_suppression,
    nms_per_class,
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
