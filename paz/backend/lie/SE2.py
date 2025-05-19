import jax.numpy as jp


def to_affine_matrix(rotation_matrix, translation):
    """Builds affine matrix from rotation matrix and translation vector.

    # Arguments
        rotation_matrix: Array (2, 2). Representing a rotation matrix.
        translation: Array (2). Translation vector.

    # Returns
        Array (4, 4) representing an affine matrix.
    """
    translation = translation.reshape(2, 1)
    affine_top = jp.concatenate([rotation_matrix, translation], axis=1)
    affine_row = jp.array([[0.0, 0.0, 1.0]])
    affine_matrix = jp.concatenate([affine_top, affine_row], axis=0)
    return affine_matrix
