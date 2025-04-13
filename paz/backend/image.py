import os
import cv2
import numpy as np
import jax
import jax.numpy as jp

import paz


BILINEAR = cv2.INTER_LINEAR
B_IMAGENET_MEAN, G_IMAGENET_MEAN, R_IMAGENET_MEAN = 104, 117, 123
BGR_IMAGENET_MEAN = (B_IMAGENET_MEAN, G_IMAGENET_MEAN, R_IMAGENET_MEAN)
RGB_IMAGENET_MEAN = (R_IMAGENET_MEAN, G_IMAGENET_MEAN, B_IMAGENET_MEAN)
rgb_IMAGENET_MEAN = jp.array(
    [
        R_IMAGENET_MEAN / 255,
        G_IMAGENET_MEAN / 255,
        B_IMAGENET_MEAN / 255,
    ]
)
B_IMAGENET_STDV, G_IMAGENET_STDV, R_IMAGENET_STDV = 57.3, 57.1, 58.4
RGB_IMAGENET_STDV = (R_IMAGENET_STDV, G_IMAGENET_STDV, B_IMAGENET_STDV)
rgb_IMAGENET_STDV = jp.array(
    [
        R_IMAGENET_STDV / 255,
        G_IMAGENET_STDV / 255,
        B_IMAGENET_STDV / 255,
    ]
)

GRAY = cv2.IMREAD_GRAYSCALE
COLOR = cv2.IMREAD_COLOR
DEPTH = cv2.IMREAD_ANYDEPTH


def flip_left_right(image):
    """Flips an image left and right.

    # Arguments
        image: Array of shape `(H, W, C)`.

    # Returns
        Flipped image array.
    """
    return image[:, ::-1]


def BGR_to_RGB(image_BGR):
    return image_BGR[..., ::-1]


def RGB_to_BGR(image_RGB):
    return image_RGB[..., ::-1]


def load(filepath, flag=COLOR):
    image = jp.array(cv2.imread(filepath, flag))
    if flag == COLOR:
        image = BGR_to_RGB(image)
    elif flag == GRAY:
        image = jp.expand_dims(image, axis=-1)
    else:
        raise ValueError("Invalid flag")
    return image


def write(filepath, image):
    image = RGB_to_BGR(image)
    image = np.ascontiguousarray(paz.to_numpy(image))
    return cv2.imwrite(filepath, image)


def resize(image, shape, method="bilinear"):
    return jax.image.resize(image, (*shape, image.shape[-1]), method)


def scale(image, scale, method="bilinear"):
    H, W, num_channels = image.shape
    H_scaled = int(H * scale[0])
    W_scaled = int(W * scale[1])
    return jax.image.resize(image, (H_scaled, W_scaled, num_channels), method)


def show(image, name="image", wait=True):
    """Shows RGB image in an external window.

    # Arguments
        image: Numpy array
        name: String indicating the window name.
        wait: Boolean. If ''True'' window stays open until user presses a key.
            If ''False'' windows closes immediately.
    """
    image = paz.to_numpy(image)
    if image.dtype != np.uint8:
        raise ValueError(f"``image`` is type {image.dtype}. Must be ``uint8``")
    image = RGB_to_BGR(image)  # openCV default color space is BGR
    cv2.imshow(name, image)
    if wait:
        while True:
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


def normalize(image):
    return image / 255.0


def denormalize(image):
    return paz.cast(image * 255.0, jp.uint8)


def rgb_to_gray(image):
    rgb_weights = jp.array([0.2989, 0.5870, 0.1140])
    grayscale = jp.tensordot(image, rgb_weights, axes=(-1, -1))
    grayscale = jp.expand_dims(grayscale, axis=-1)
    return grayscale


def RGB_to_GRAY(image):
    image = normalize(image)
    image = rgb_to_gray(image)
    image = denormalize(image)
    return image


def preprocess(image, shape):
    return normalize(resize(image, shape))


def get_dimensions(image):
    H, W = image.shape[:2]
    return H, W


def crop(image, box):
    x_min, y_min, x_max, y_max = box
    return image[y_min:y_max, x_min:x_max, :]


def crop_center(image, crop_shape):
    H_new, W_new = crop_shape
    H_now, W_now = get_dimensions(image)
    center_x = W_now // 2
    center_y = H_now // 2
    x_min = center_x - (W_new // 2)
    y_min = center_y - (H_new // 2)
    x_max = x_min + W_new
    y_max = y_min + H_new
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image


def standardize(image, mean, stdv):
    return (image - mean) / stdv


def normalize_min_max(x, axis=-1):
    x_min = x.min(axis=axis, keepdims=True)
    x_max = x.max(axis=axis, keepdims=True)
    return (x - x_min) / (x_max - x_min)


def random_brightness(key, image, delta=32):
    """Applies random brightness to an RGB image.

    # Arguments
        image: Numpy array representing an image RGB format.
        delta: Int.
    """
    image = paz.cast(image, jp.float32)
    random_brightness = jax.random.uniform(key, (), jp.float32, -delta, delta)
    image = image + random_brightness
    image = jp.clip(image, 0, 255)
    image = paz.cast(image, jp.uint8)
    return image


def random_contrast(key, image, lower=0.5, upper=1.5):
    """Applies random contrast to an RGB image.

    # Arguments
        image: Numpy array representing an image RGB format.
        lower: Float.
        upper: Float.
    """
    alpha = jax.random.uniform(key, (1,), jp.float32, lower, upper)
    image = paz.cast(image, jp.float32)
    image = image * alpha
    image = jp.clip(image, 0, 255)
    image = paz.cast(image, jp.uint8)
    return image


def split_channels(image, num_channels=3):
    channels = jp.split(image, num_channels, axis=-1)
    return tuple(jp.squeeze(channel, axis=-1) for channel in channels)


def merge_channels(channel_0, channel_1, channel_2):
    channel_0 = jp.expand_dims(channel_0, axis=-1)
    channel_1 = jp.expand_dims(channel_1, axis=-1)
    channel_2 = jp.expand_dims(channel_2, axis=-1)
    return jp.concatenate([channel_0, channel_1, channel_2], axis=-1)


def rgb_to_hsv(image):
    """Convert image from RGB to HSV."""
    r, g, b = split_channels(image)
    channels_max = jp.max(image, axis=-1)  # value = channels_max
    channels_min = jp.min(image, axis=-1)
    delta = channels_max - channels_min
    safe_value = jp.where(channels_max > 0, channels_max, 1.0)
    safe_delta = jp.where(delta > 0, delta, 1.0)
    saturation = jp.where(channels_max > 0, delta / safe_value, 0.0)
    norm = 1.0 / (6.0 * safe_delta)
    hue = jp.where(
        channels_max == g,
        norm * (b - r) + 2.0 / 6.0,
        norm * (r - g) + 4.0 / 6.0,
    )
    hue = jp.where(channels_max == r, norm * (g - b), hue)
    hue = jp.where(delta > 0, hue, 0.0) + (hue < 0.0)
    return merge_channels(hue, saturation, channels_max)


def hsv_to_rgb(image):
    """Converts hue, saturation, value planes to r, g, b color planes."""
    hue, saturation, value = split_channels(image)
    dh = (hue % 1.0) * 6.0  # Wrap when hue >= 360°.
    dr = jp.clip(jp.abs(dh - 3.0) - 1.0, 0.0, 1.0)
    dg = jp.clip(2.0 - jp.abs(dh - 2.0), 0.0, 1.0)
    db = jp.clip(2.0 - jp.abs(dh - 4.0), 0.0, 1.0)
    one_minus_saturation = 1.0 - saturation
    r = value * (one_minus_saturation + saturation * dr)
    g = value * (one_minus_saturation + saturation * dg)
    b = value * (one_minus_saturation + saturation * db)
    return merge_channels(r, g, b)


def random_saturation(key, image, lower=0.3, upper=1.5):
    """Applies random saturation to an RGB image."""
    image = normalize(image)
    image = rgb_to_hsv(image)
    h, s, v = split_channels(image)
    random_scale = jax.random.uniform(key, (), jp.float32, lower, upper)
    s = s * random_scale
    s = np.clip(s, 0.0, 1.0)
    image = merge_channels(h, s, v)
    image = hsv_to_rgb(image)
    image = denormalize(image)
    return paz.cast(image, jp.uint8)
