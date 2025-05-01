from jax.scipy.ndimage import map_coordinates
import jax.numpy as jp
import paz


def affine_transform(image, matrix, order=1, mode="nearest", cval=0.0):

    def build_image_indices(image):
        dimension_indices = [jp.arange(size) for size in image.shape]
        meshgrid = jp.meshgrid(*dimension_indices, indexing="ij")
        meshgrid = [jp.expand_dims(x, axis=-1) for x in meshgrid]
        indices = jp.concatenate(meshgrid, axis=-1)
        return indices

    offset = matrix[: image.ndim, image.ndim]
    matrix = matrix[: image.ndim, : image.ndim]
    coordinates = build_image_indices(image) @ matrix.T
    coordinates = jp.moveaxis(coordinates, source=-1, destination=0)
    offset = jp.full((3,), fill_value=offset)
    coordinates += jp.reshape(offset, (*offset.shape, 1, 1, 1))
    return map_coordinates(image, coordinates, order, mode, cval)


def rotate(image, angle, order=1, mode="nearest", cval=0.0):
    """Rotates an image around its center using interpolation."""
    rotation = paz.SO3.rotation_z(angle)
    image_center = (jp.asarray(image.shape) - 1.0) / 2.0
    translation = image_center - rotation @ image_center
    matrix = paz.SE3.to_affine_matrix(rotation, translation)
    return affine_transform(image, matrix, order=order, mode=mode, cval=cval)
