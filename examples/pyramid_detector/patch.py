import jax
import jax.numpy as jp


def boxes_patch(H, W, patch_size, strides):
    H_patch, W_patch = patch_size
    y_stride, x_stride = strides

    y_min_args = jp.arange(0, H - H_patch + 1, y_stride)
    x_min_args = jp.arange(0, W - W_patch + 1, x_stride)

    def compute_box(y_min, x_min):
        y_max = y_min + H_patch
        x_max = x_min + W_patch
        return jp.array([x_min, y_min, x_max, y_max], dtype=jp.int32)

    def compute_row_boxes(y_min_args):
        return jax.vmap(compute_box, in_axes=(None, 0))(y_min_args, x_min_args)

    return jax.vmap(compute_row_boxes)(y_min_args)


def get_patch_boxes_same(H, W, patch_size, stride):
    H_patch, W_patch = patch_size
    y_stride, x_stride = stride

    H_out = (H + y_stride - 1) // y_stride
    W_out = (W + x_stride - 1) // x_stride

    y_needed_pad = max(0, (H_out - 1) * y_stride + H_patch - H)
    pad_top = y_needed_pad // 2

    x_needed_pad = max(0, (W_out - 1) * x_stride + W_patch - W)
    pad_left = x_needed_pad // 2

    y_starts_on_padded_grid = jp.arange(H_out) * y_stride
    x_starts_on_padded_grid = jp.arange(W_out) * x_stride

    y_min_coords_in_original_ref = y_starts_on_padded_grid - pad_top
    x_min_coords_in_original_ref = x_starts_on_padded_grid - pad_left

    def compute_single_box(y_start_orig, x_start_orig):
        y_min = y_start_orig
        x_min = x_start_orig
        y_max = y_start_orig + H_patch
        x_max = x_start_orig + W_patch
        return jp.array([x_min, y_min, x_max, y_max], dtype=jp.int32)

    def compute_row_of_boxes(single_y_min_original_coord):
        return jax.vmap(compute_single_box, in_axes=(None, 0), out_axes=0)(
            single_y_min_original_coord, x_min_coords_in_original_ref
        )

    return jax.vmap(compute_row_of_boxes, in_axes=(0), out_axes=0)(
        y_min_coords_in_original_ref
    )


if __name__ == "__main__":
    import paz

    from deepfish import load

    train_images, train_labels = load("Deepfish/", "train")
    image = paz.image.load(train_images[0])
    H = 480
    W = 640
    patch_size = (128, 128)
    strides = (128, 128)
    padding = "same"
    image = paz.image.resize(image, (H, W))

    patches = paz.image.patch(image, patch_size, strides, padding)

    if padding == "valid":
        boxes = boxes_patch(H, W, patch_size, strides)
    else:
        boxes = get_patch_boxes_same(H, W, patch_size, strides)
    num_row, num_patch_cols, H_patch, W_patch, C = patches.shape
    paz.image.show(
        paz.draw.mosaic(
            patches.reshape(-1, *patch_size, 3).astype("uint8"),
            (num_row, num_patch_cols),
            border=2,
        ).astype("uint8")
    )

    boxes = boxes.reshape(-1, 4)
    print(patches.shape)
    paz.image.show(paz.draw.boxes(image.astype("uint8"), boxes))
