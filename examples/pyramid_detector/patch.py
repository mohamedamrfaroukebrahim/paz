import jax
import jax.numpy as jp


def get_patch_shape(H, W, patch_size, strides, padding="valid"):
    H_patch, W_patch = patch_size
    y_stride, x_stride = strides
    if padding == "same":
        num_patch_rows = (H + y_stride - 1) // y_stride
        num_patch_cols = (W + x_stride - 1) // x_stride
    elif padding == "valid":
        num_patch_rows = (H - H_patch) // y_stride + 1
        num_patch_cols = (W - W_patch) // x_stride + 1
    else:
        raise ValueError(f"Unknown padding type: {padding}")
    return num_patch_rows, num_patch_cols


def image_pad_same(image, patch_size, strides):

    def get_patch_span(num_patches, stride_size, patch_size):
        num_stride_steps = num_patches - 1
        stride_distance = num_stride_steps * stride_size
        total_covered_distance = stride_distance + patch_size
        return total_covered_distance

    def compute_cover(H, W, patch_size, strides):
        patch_shape = get_patch_shape(H, W, patch_size, strides, "same")
        H_covered = get_patch_span(patch_shape[0], strides[0], patch_size[0])
        W_covered = get_patch_span(patch_shape[1], strides[1], patch_size[1])
        return H_covered, W_covered

    def compute_needed_pad(covered_size, original_size):
        total_residue = covered_size - original_size
        minor_half_residue = total_residue // 2
        major_half_residue = total_residue - minor_half_residue
        return minor_half_residue, major_half_residue

    H, W = paz.image.get_size(image)
    H_covered, W_covered = compute_cover(H, W, patch_size, strides)
    y_minor_pad, y_major_pad = compute_needed_pad(H_covered, H)
    x_minor_pad, x_major_pad = compute_needed_pad(W_covered, W)
    pad_sizes = (y_minor_pad, y_major_pad, x_minor_pad, x_major_pad)
    return paz.image.pad(image, *pad_sizes)


def patch(image, patch_size, strides, padding="valid"):

    def build_min_args(H, W, patch_size, strides):
        H_patch, W_patch = patch_size
        y_stride, x_stride = strides
        y_min_args = jp.arange(H) * y_stride
        x_min_args = jp.arange(W) * x_stride
        return y_min_args, x_min_args

    def vectorized_patch(image, y_min_args, x_min_args, patch_size):
        H_patch, W_patch = patch_size

        def patch_one(y_min_args, x_min_args):
            start_args = (y_min_args, x_min_args, 0)
            slice_args = (H_patch, W_patch, paz.image.num_channels(image))
            return jax.lax.dynamic_slice(image, start_args, slice_args)

        def patch_rows(y_min_args):
            return jax.vmap(patch_one, (None, 0))(y_min_args, x_min_args)

        return jax.vmap(patch_rows)(y_min_args)

    H, W = paz.image.get_size(image)
    H_out, W_out = get_patch_shape(H, W, patch_size, strides, padding)
    y_min_args, x_min_args = build_min_args(H_out, W_out, patch_size, strides)
    if padding == "same":
        image = image_pad_same(image, patch_size, strides)
    return vectorized_patch(image, y_min_args, x_min_args, patch_size)


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
    strides = (32, 32)
    padding = "same"
    image = paz.image.resize(image, (H, W))

    patches = patch(image, patch_size, strides, padding)

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
