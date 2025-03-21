from collections import namedtuple
import jax.numpy as jp
import cv2
import paz


def is_open(camera):
    """Checks if camera is open.

    # Returns
        Boolean
    """
    return camera._camera.isOpened()


def is_close(camera):
    """Checks if camera is close.

    # Returns
        Boolean
    """
    return not is_open(camera)


def read(camera):
    """Reads camera input and returns a frame.

    # Returns
        Image array.
    """
    return camera._camera.read()[1]


def stop(camera):
    """Stops capturing device."""
    return camera._camera.release()


def start(camera):
    """Starts capturing device

    # Returns
        Camera object.
    """
    camera._camera = cv2.VideoCapture(camera.indentifier)
    if (camera._camera is None) or is_close(camera):
        raise ValueError("Unable to open device", camera.identifier)
    return camera._camera


def take_photo(camera):
    """Starts camera, reads buffer and returns an image.

    # Arguments:
        camera: paz.Camera namedtuple.

    # Returns:
        Image array.
    """
    start(camera)
    image = paz.image.BGR_to_RGB(read(camera))
    stop(camera)
    return image


def compute_focal_length(W, HFOV):
    return (W / 2) * (1 / jp.tan(jp.deg2rad(HFOV / 2.0)))


def intrinsics_from_HFOV(camera, image_shape=None, HFOV=70):
    """Computes camera intrinsics using horizontal field of view (HFOV).

    # Arguments
        HFOV: Angle in degrees of horizontal field of view.
        image_shape: List of two floats [H, W].

    # Returns
        camera intrinsics array (3, 3).

    # Notes:

                   \           /      ^
                    \         /       |
                     \ lens  /        | w/2
    horizontal field  \     / alpha/2 |
    of view (alpha)____\( )/_________ |      image
                       /( )\          |      plane
                      /     <-- f --> |
                     /       \        |
                    /         \       |
                   /           \      v

                Pinhole camera model

    From the image above we know that: tan(alpha/2) = w/2f
    -> f = w/2 * (1/tan(alpha/2))

    alpha in webcams and phones is often between 50 and 70 degrees.
    -> 0.7 w <= f <= w
    """
    if image_shape is None:
        start(camera)
        H, W = paz.image.get_dimensions(read(camera))
        stop(camera)
    else:
        H, W = image_shape[:2]

    focal_length = compute_focal_length(W, HFOV)
    camera_intrinsics = jp.array(
        [
            [focal_length, 0, W / 2.0],
            [0, focal_length, H / 2.0],
            [0, 0, 1.0],
        ]
    )
    return camera_intrinsics


# State = namedtuple(
#     "State", ["identifier", "name", "intrinsics", "distortion", "_camera"]
# )


class Camera(object):
    """Camera abstract class."""

    def __init__(
        self, indentifier=0, name="Camera", intrinsics=None, distortion=None
    ):
        self.identifier = identifier
        self.name = name
        self.intrinsics = intrinsics
        self.distortion = None
        self._camera = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def intrinsics(self):
        return self._intrinsics

    @intrinsics.setter
    def intrinsics(self, value):
        if value is None:
            value = jp.zeros((4))
        self._intrinsics = value

    @property
    def distortion(self):
        return self._distortion

    @distortion.setter
    def distortion(self, distortion):
        self._distortion = distortion

    def is_open(self):
        return self._camera.isOpened()

    def is_close(self):
        return not self.is_open()

    def start(self):
        self._camera = cv2.VideoCapture(self.indentifier)
        if (self._camera is None) or self.is_close():
            raise ValueError("Unable to open device", self.identifier)

    def stop(self):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError

    def save(self, filepath):
        raise NotImplementedError

    def load(self, filepath):
        raise NotImplementedError

    def intrinsics_from_HFOV(self, HFOV=70, image_shape=None):
        raise NotImplementedError

    def take_photo(self):
        raise NotImplementedError
