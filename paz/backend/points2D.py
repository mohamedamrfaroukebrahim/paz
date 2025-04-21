import jax.numpy as jp


def add_ones(points2D):
    ones = jp.ones((len(points2D), 1))
    points2D = jp.concatenate([points2D, ones], axis=-1)
    return points2D


def transform(points2D, affine_transform):
    original_dimension = points2D.shape[-1]
    points2D = add_ones(points2D)
    points2D = jp.matmul(affine_transform, points2D.T).T
    return points2D[:, :original_dimension]
