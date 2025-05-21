import jax
import jax.numpy as jp
import paz


def slide_window_patches(
    image: jax.Array, window_size: tuple[int, int], stride: tuple[int, int]
) -> tuple[jax.Array, jax.Array]:
    """
    Extracts sliding window patches from a single image using JAX.
    Ensures input to convolution is float32.
    """
    img_h, img_w, img_c = image.shape
    win_h, win_w = window_size
    stride_y, stride_x = stride

    if win_h > img_h or win_w > img_w:
        return jp.empty((0, win_h, win_w, img_c), dtype=jp.float32), jp.empty(
            (0, 4), dtype=jp.int32
        )

    # --- KEY CHANGE: Convert image to float32 before convolution ---
    image_for_conv = image.astype(jp.float32)
    # -----------------------------------------------------------------

    image_batched = image_for_conv[None, ...]  # Shape: (1, img_H, img_W, img_C)

    patches_flat = jax.lax.conv_general_dilated_patches(
        lhs=image_batched,  # Now definitely float32
        filter_shape=window_size,  # Corrected keyword from previous discussion
        window_strides=stride,
        padding="VALID",
        dimension_numbers=("NHWC", "OIHW", "NHWC"),
    )
    # Output shape of patches_flat: (1, num_windows_y, num_windows_x, win_h * win_w * img_c)
    # The dtype of patches_flat will now also be float32.

    _, num_windows_y, num_windows_x, _ = patches_flat.shape

    if num_windows_y == 0 or num_windows_x == 0:
        return jp.empty((0, win_h, win_w, img_c), dtype=jp.float32), jp.empty(
            (0, 4), dtype=jp.int32
        )

    num_total_patches = num_windows_y * num_windows_x
    # Patches will be float32
    patches = patches_flat.reshape((num_total_patches, win_h, win_w, img_c))

    y_starts = jp.arange(num_windows_y) * stride_y
    x_starts = jp.arange(num_windows_x) * stride_x

    grid_y, grid_x = jp.meshgrid(y_starts, x_starts, indexing="ij")
    flat_y_starts = grid_y.reshape(-1)
    flat_x_starts = grid_x.reshape(-1)

    coordinates = jp.stack(
        [
            flat_x_starts,
            flat_y_starts,
            flat_x_starts + win_w,
            flat_y_starts + win_h,
        ],
        axis=-1,
    ).astype(jp.int32)

    return patches, coordinates


def detect(image, classify, scales, window_size, stride, score_threshold=0.5):
    pyramid_levels = paz.image.pyramid(image, scales)
    pyramid_detections, pyramid_scores = [], []
    for scaled_image, scale in zip(pyramid_levels, scales):
        patches_batch, local_coordinates_batch = slide_window_patches(
            scaled_image, window_size, stride
        )
        logits = classify(patches_batch, training=False)
        scores = jax.nn.sigmoid(logits[:, 0])
        positive_args = jp.where(scores > score_threshold)[0]

        if positive_args.shape[0] > 0:
            positive_scores = scores[positive_args]
            positive_local_coords = local_coordinates_batch[positive_args]

            original_coords_at_level = (
                positive_local_coords.astype(jp.float32) / scale
            )
            pyramid_detections.append(original_coords_at_level)
            pyramid_scores.append(positive_scores)

    if not pyramid_detections:
        return jp.empty((0, 5), dtype=jp.float32)

    detections = jp.concatenate(pyramid_detections, axis=0)
    scores = jp.concatenate(pyramid_scores, axis=0)
    return paz.detection.join(detections, scores)


if __name__ == "__main__":
    import os

    os.environ["KERAS_BACKEND"] = "jax"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
    from deepfish import load

    train_images, train_labels = load("Deepfish/", "train")
    image = paz.image.load(train_images[0])
    image = paz.image.resize(image, (480, 640))
    print(image.dtype, image.shape)
    # patches, coords = slide_window_patches(image, (128, 128), (32, 32))
    patches = jax.lax.conv_general_dilated_patches(
        lhs=image[None, ...],
        filter_shape=(128, 128),
        window_strides=(32, 32),
        padding="VALID",
        dimension_numbers=("NHWC", "OIHW", "NHWC"),
    )
