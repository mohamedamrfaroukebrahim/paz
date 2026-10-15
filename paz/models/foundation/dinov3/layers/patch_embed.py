from keras import Input, Model
from keras.layers import Conv2D, Reshape


def make_2tuple(x):
    if isinstance(x, (list, tuple)):
        assert len(x) == 2, f"Expected tuple of length 2, but got {len(x)}"
        return tuple(x)
    assert isinstance(x, int), f"Expected int or tuple, but got {type(x)}"
    return (x, x)


def PatchEmbed(img_size=224, patch_size=16, in_channels=3, embed_dim=768,
               norm_layer=None, flatten_embedding=True, use_bias=True,
               name="patch_embed"):
    """Image to patch tokens, channels-last ``(B, H, W, C) -> (B, N, D)``.

    ``norm_layer`` is a callable taking ``name=`` that returns a layer.
    """
    image_shape = make_2tuple(img_size) + (in_channels,)
    images = Input(image_shape, name=f"{name}_images")
    args = (images, make_2tuple(patch_size), embed_dim, use_bias, name)
    patches = apply_patch_embed(*args)
    height, width = patches.shape[1], patches.shape[2]
    tokens = flatten_patches(patches, embed_dim, name)
    if norm_layer is not None:
        tokens = norm_layer(name=f"{name}_norm")(tokens)
    if not flatten_embedding:
        shape = (height, width, embed_dim)
        tokens = Reshape(shape, name=f"{name}_unflatten")(tokens)
    return Model(images, tokens, name=name)


def apply_patch_embed(images, patch_size, embed_dim, use_bias, name):
    kwargs = dict(strides=patch_size, padding="valid", use_bias=use_bias)
    projection = Conv2D(embed_dim, patch_size, name=f"{name}_proj", **kwargs)
    return projection(images)


def flatten_patches(patches, embed_dim, name):
    num_patches = patches.shape[1] * patches.shape[2]
    shape = (num_patches, embed_dim)
    return Reshape(shape, name=f"{name}_flatten")(patches)
