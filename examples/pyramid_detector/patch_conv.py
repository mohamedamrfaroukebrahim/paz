import jax
import jax.numpy as jp


def patch(image, window_size, stride):
    img_h, img_w, img_c = image.shape
    win_h, win_w = window_size
    stride_y, stride_x = stride

    # image_for_conv = image.astype(jp.float32)

    image_batched = image[None, ...]  # Shape: (1, img_H, img_W, img_C)

    patches_flat = jax.lax.conv_general_dilated_patches(
        lhs=image_batched,  # Now definitely float32
        filter_shape=window_size,  # Corrected keyword from previous discussion
        window_strides=stride,
        padding="VALID",
        dimension_numbers=("NHWC", "OIHW", "NHWC"),
    )

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


if __name__ == "__main__":
    import paz

    from deepfish import load

    train_images, train_labels = load("Deepfish/", "train")
    image = paz.image.load(train_images[0])
    image = paz.image.resize(image, (480, 640))
    print(image.dtype, image.shape)
    size = 128
    patches, coordinates = jax.jit(
        paz.partial(patch, window_size=(size, size), stride=(32, 32))
    )(image)
    # patches, coordinates = patch(image, (128, 128), (128, 128))
    paz.image.show(
        paz.draw.mosaic(
            patches.reshape(-1, size, size, 3).astype("uint8"),
            (15, 20),
            border=2,
        ).astype("uint8")
    )
