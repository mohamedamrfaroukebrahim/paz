import jax
import jax.numpy as jp


def create_image_pyramid_jax(image, scale_factor, min_size):
    if not (0 < scale_factor < 1):
        raise ValueError("Scale factor must be between 0 and 1.")

    pyramid = []
    current_scale = 1.0
    current_image = image
    min_h, min_w = min_size

    while True:
        img_h, img_w = current_image.shape[:2]

        # Stop if the current image is smaller than the minimum size
        if img_h < min_h or img_w < min_w:
            break

        pyramid.append((current_image, current_scale))

        # Calculate new dimensions for the next level
        # Round to nearest integer and ensure dimensions are at least 1
        new_h = max(1, int(round(img_h * scale_factor)))
        new_w = max(1, int(round(img_w * scale_factor)))

        # Stop if the new dimensions are too small to be practical or smaller than min_size
        # (The primary check at the loop start handles min_size effectively)
        if new_h < 1 or new_w < 1 or new_h < min_h or new_w < min_w:
            # Secondary check to ensure we don't go significantly below min_size due to rounding
            # or create degenerate images. If new_h/w is already < min_h/w, we might stop
            # based on the loop condition, but this adds robustness.
            # More simply, if new_h/w are too small to be useful (e.g. smaller than a patch)
            if (
                new_h < min_size[0] or new_w < min_size[1]
            ):  # ensure we don't add then break
                if not (
                    img_h == new_h and img_w == new_w
                ):  # if scale factor caused no change and we're small
                    break

        if (
            img_h == new_h and img_w == new_w and current_scale != 1.0
        ):  # No change in size, avoid infinite loop
            break

        # Resize the image for the next level
        # jax.image.resize expects the output shape as (new_H, new_W, C)
        current_image = jax.image.resize(
            current_image,
            (new_h, new_w, current_image.shape[-1]),
            method="bilinear",  # Common methods: 'bilinear', 'nearest', 'bicubic'
        )
        current_scale *= scale_factor

    return pyramid
