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


if __name__ == "__main__":
    import paz

    from deepfish import load

    train_images, train_labels = load("Deepfish/", "train")
    image = paz.image.load(train_images[0])
    image = paz.image.resize(image, (480, 640))
    print(image.dtype, image.shape)
    patches = to_patches(image, (32, 32), (32, 32))
    paz.image.show(
        paz.draw.mosaic(
            patches.reshape(-1, 32, 32, 3).astype("uint8"),
            (15, 20),
            border=2,
        ).astype("uint8")
    )
