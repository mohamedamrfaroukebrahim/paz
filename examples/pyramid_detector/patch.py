import jax
import jax.numpy as jp


def boxes_patch(H, W, patch_size, strides, padding="valid"):
    H_patch, W_patch = patch_size
    y_stride, x_stride = strides

    H_out, W_out = paz.image.get_patch_shape(H, W, patch_size, strides, padding)
    y_min_args = jp.arange(H_out) * y_stride
    x_min_args = jp.arange(W_out) * x_stride

    if padding == "same":
        y_needed_pad = max(0, (H_out - 1) * y_stride + H_patch - H)
        x_needed_pad = max(0, (W_out - 1) * x_stride + W_patch - W)
        y_minor_half_residue = y_needed_pad // 2
        x_minor_half_residue = x_needed_pad // 2
        y_min_args = y_min_args - y_minor_half_residue  # pad top
        x_min_args = x_min_args - x_minor_half_residue  # pad left

    def make_one_box(y_min, x_min):
        y_max = y_min + H_patch
        x_max = x_min + W_patch
        return jp.array([x_min, y_min, x_max, y_max], dtype=jp.int32)

    def make_box_rows(y_min_args):
        return jax.vmap(make_one_box, in_axes=(None, 0))(y_min_args, x_min_args)

    return jax.vmap(make_box_rows)(y_min_args)


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

    patches = paz.image.patch(image, patch_size, strides, padding)
    boxes = boxes_patch(H, W, patch_size, strides, padding)
    num_row, num_patch_cols, H_patch, W_patch, C = patches.shape
    paz.image.show(
        paz.draw.mosaic(
            patches.reshape(-1, *patch_size, 3).astype("uint8"),
            (num_row, num_patch_cols),
            border=2,
        ).astype("uint8")
    )

    boxes = boxes.reshape(-1, 4)
    # for box in boxes:
    #     paz.image.show(paz.draw.boxes(image.astype("uint8"), box[None, :]))
    paz.image.show(paz.draw.boxes(image.astype("uint8"), boxes))
