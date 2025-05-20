import pytest
import jax.numpy as jp
from match import match
from prior_boxes import create_prior_boxes


@pytest.fixture
def boxes_with_label():
    box_with_label = jp.array(
        [
            [47.0, 239.0, 194.0, 370.0, 12.0],
            [7.0, 11.0, 351.0, 497.0, 15.0],
            [138.0, 199.0, 206.0, 300.0, 19.0],
            [122.0, 154.0, 214.0, 194.0, 18.0],
            [238.0, 155.0, 306.0, 204.0, 9.0],
        ]
    )
    return box_with_label


@pytest.fixture
def target_unique_matches():
    return jp.array([[47.0, 239.0, 194.0, 370.0], [238.0, 155.0, 306.0, 204.0]])


def test_match_box(boxes_with_label, target_unique_matches):
    matched_boxes = match(boxes_with_label, create_prior_boxes("VOC"))
    assert jp.array_equal(
        target_unique_matches, jp.unique(matched_boxes[:, :-1], axis=0)
    )
