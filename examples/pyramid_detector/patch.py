import jax
import jax.numpy as jp


def _compute_output_shape(H, W, patch_size, strides):
    H_patch, W_patch = patch_size
    y_stride, x_stride = strides
    H_out = (H + y_stride - 1) // y_stride
    W_out = (W + x_stride - 1) // x_stride
    return H_out, W_out


def _compute_padding(H, W, patch_size, strides):
    H_out, W_out = _compute_output_shape(H, W, patch_size, strides)
    H_patch, W_patch = patch_size
    y_stride, x_stride = strides

    effective_H_covered = (H_out - 1) * y_stride + H_patch
    effective_W_covered = (W_out - 1) * x_stride + W_patch

    y_needed_pad = max(0, effective_H_covered - H)
    pad_top = y_needed_pad // 2
    pad_bottom = y_needed_pad - pad_top

    x_needed_pad = max(0, effective_W_covered - W)
    pad_left = x_needed_pad // 2
    pad_right = x_needed_pad - pad_left
    return pad_top, pad_bottom, pad_left, pad_right


def _patch(image, y_min_args, x_min_args, patch_size):
    H_patch, W_patch = patch_size

    def patch_one(y_min_args, x_min_args):
        start_args = (y_min_args, x_min_args, 0)
        slice_args = (H_patch, W_patch, paz.image.num_channels(image))
        return jax.lax.dynamic_slice(image, start_args, slice_args)

    def patch_rows(y_min_args):
        return jax.vmap(patch_one, (None, 0))(y_min_args, x_min_args)

    return jax.vmap(patch_rows)(y_min_args)


def _build_min_args(H, W, strides):
    y_stride, x_stride = strides
    y_min_args = jp.arange(H) * y_stride
    x_min_args = jp.arange(W) * x_stride
    return y_min_args, x_min_args


def patch_same(image, patch_size, strides):
    H, W = paz.image.get_size(image)
    H_out, W_out = _compute_output_shape(H, W, patch_size, strides)
    pad_widths = _compute_padding(H, W, patch_size, strides)
    padded_image = paz.image.pad(image, *pad_widths)
    y_min_args, x_min_args = _build_min_args(H_out, W_out, strides)

    return _patch(padded_image, y_min_args, x_min_args, patch_size)


def patch(image, patch_size, strides):
    H, W, C = image.shape
    y_stride, x_stride = strides
    H_patch, W_patch = patch_size
    y_min_args = jp.arange(0, H - H_patch + 1, y_stride)
    x_min_args = jp.arange(0, W - W_patch + 1, x_stride)

    return _patch(image, y_min_args, x_min_args, patch_size)


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
    window_size = 128
    stride = 128
    padding = "same"
    image = paz.image.resize(image, (H, W))
    print(image.dtype, image.shape)

    if padding == "valid":
        patches = patch(image, (window_size, window_size), (stride, stride))
        boxes = boxes_patch(H, W, (window_size, window_size), (stride, stride))
    else:
        patches = patch_same(
            image, (window_size, window_size), (stride, stride)
        )
        boxes = get_patch_boxes_same(
            H, W, (window_size, window_size), (stride, stride)
        )
    num_row, num_cols, H_patch, W_patch, C = patches.shape
    paz.image.show(
        paz.draw.mosaic(
            patches.reshape(-1, window_size, window_size, 3).astype("uint8"),
            (num_row, num_cols),
            border=2,
        ).astype("uint8")
    )

    boxes = boxes.reshape(-1, 4)
    paz.image.show(paz.draw.boxes(image.astype("uint8"), boxes))
