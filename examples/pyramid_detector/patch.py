import jax
import jax.numpy as jnp


def to_patches(image, patch_size, stride):
    H, W, C = image.shape
    H_patch, W_patch = patch_size
    y_stride, x_stride = stride

    y_start = jnp.arange(0, H - H_patch + 1, y_stride)
    x_start = jnp.arange(0, W - W_patch + 1, x_stride)
    print(x_start, y_start)

    def get_single_patch(y_start_index, x_start_index):
        return jax.lax.dynamic_slice(
            image,
            (y_start_index, x_start_index, 0),
            (H_patch, W_patch, C),
        )

    def extract_row_of_patches(single_y_coord):
        return jax.vmap(get_single_patch, in_axes=(None, 0), out_axes=0)(
            single_y_coord, x_start
        )

    return jax.vmap(extract_row_of_patches, in_axes=0, out_axes=0)(y_start)


def get_patch_boxes(H, W, patch_size, stride):
    H_patch, W_patch = patch_size
    y_stride, x_stride = stride

    y_starts_coords = jnp.arange(0, H - H_patch + 1, y_stride)
    x_starts_coords = jnp.arange(0, W - W_patch + 1, x_stride)

    def compute_box(y_start_coord, x_start_coord):
        y_min = y_start_coord
        x_min = x_start_coord
        y_max = y_start_coord + H_patch
        x_max = x_start_coord + W_patch
        return jnp.array([y_min, x_min, y_max, x_max], dtype=jnp.int32)

    def compute_row_boxes(single_y_coord):
        return jax.vmap(compute_box, in_axes=(None, 0), out_axes=0)(
            single_y_coord, x_starts_coords
        )

    # Create boxes for all rows (vmap over y_starts)
    return jax.vmap(compute_row_boxes, in_axes=(0), out_axes=0)(y_starts_coords)


if __name__ == "__main__":
    import paz

    from deepfish import load

    train_images, train_labels = load("Deepfish/", "train")
    image = paz.image.load(train_images[0])
    H = 480
    W = 640
    window_size = 128
    stride = 32
    image = paz.image.resize(image, (H, W))
    print(image.dtype, image.shape)
    patches = to_patches(image, (window_size, window_size), (stride, stride))
    num_row, num_cols, H_patch, W_patch, C = patches.shape
    paz.image.show(
        paz.draw.mosaic(
            patches.reshape(-1, window_size, window_size, 3).astype("uint8"),
            (num_row, num_cols),
            border=2,
        ).astype("uint8")
    )
    boxes = get_patch_boxes(H, W, (window_size, window_size), (stride, stride))
