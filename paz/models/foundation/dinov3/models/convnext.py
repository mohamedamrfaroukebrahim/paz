from functools import partial

import numpy as np
from keras import Input, Model, ops
from keras.layers import Activation, Add, Concatenate, Conv2D, Dense
from keras.layers import DepthwiseConv2D, Dropout, EinsumDense
from keras.layers import GlobalAveragePooling2D
from keras.layers import LayerNormalization, Permute, Reshape

from paz.models.foundation.dinov3.layers.layer_scale import LayerScale


def ConvNeXt(in_chans=3, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768),
             drop_path_rate=0.0, layer_scale_init_value=1e-6, img_size=None,
             name="convnext", **ignored_kwargs):
    """DINOv3 ConvNeXt returning ``(class_token, patch_tokens)``.

    Takes channels-first images. Layer names mirror the PyTorch state-dict
    keys with dots replaced by underscores, e.g. ``stages_0_1_dwconv``.
    """
    if len(ignored_kwargs) > 0:
        print(f"Ignored kwargs: {ignored_kwargs}")
    images = Input((in_chans, img_size, img_size), name="images")
    x = apply_stem(images, dims[0])
    drop_rates = np.linspace(0.0, drop_path_rate, sum(depths)).tolist()
    block_index = 0
    for stage, (depth, dim) in enumerate(zip(depths, dims)):
        if stage > 0:
            x = apply_downsample(x, dim, f"downsample_layers_{stage}")
        for index in range(depth):
            args = (dim, drop_rates[block_index], layer_scale_init_value)
            x = apply_block(x, *args, f"stages_{stage}_{index}")
            block_index = block_index + 1
    outputs = pool_and_normalize(x, dims[-1])
    return Model(images, outputs, name=name)


def LayerNorm(dim, data_format="channels_last", epsilon=1e-6, name=None):
    """ConvNeXt layer norm as a model, weights ``[kernel, bias]`` of (dim,).

    Computes ``(x - mean) / sqrt(variance + epsilon)`` then the affine, the
    arithmetic of the PyTorch and pre-refactor norms, so outputs match them
    exactly; ``LayerNormalization`` rounds differently.
    """
    channels_first = data_format == "channels_first"
    axis = 1 if channels_first else -1
    shape = (dim, None, None) if channels_first else (None, None, dim)
    inputs = Input(shape, name=f"{name}_inputs")
    mean = ops.mean(inputs, axis=axis, keepdims=True)
    variance = ops.mean(ops.square(inputs - mean), axis=axis, keepdims=True)
    x = (inputs - mean) / ops.sqrt(variance + epsilon)
    if channels_first:
        x = Permute((2, 3, 1), name=f"{name}_to_channels_last")(x)
    kwargs = dict(output_shape=(dim,), bias_axes="d", name=f"{name}_affine")
    kwargs["kernel_initializer"] = "ones"
    x = EinsumDense("...d,d->...d", **kwargs)(x)
    if channels_first:
        x = Permute((3, 1, 2), name=f"{name}_to_channels_first")(x)
    return Model(inputs, x, name=name)


def apply_stem(x, dim):
    kwargs = dict(strides=4, data_format="channels_first")
    x = Conv2D(dim, 4, **kwargs, name="downsample_layers_0_0")(x)
    return LayerNorm(dim, "channels_first", name="downsample_layers_0_1")(x)


def apply_downsample(x, dim, name):
    x = LayerNorm(x.shape[1], "channels_first", name=f"{name}_0")(x)
    kwargs = dict(strides=2, data_format="channels_first")
    return Conv2D(dim, 2, **kwargs, name=f"{name}_1")(x)


def apply_block(x, dim, drop_path, layer_scale_init_value, name):
    """Depthwise conv then a pointwise MLP, added back to the input."""
    shortcut = x
    kwargs = dict(padding="same", data_format="channels_first")
    x = DepthwiseConv2D(7, **kwargs, name=f"{name}_dwconv")(x)
    x = Permute((2, 3, 1), name=f"{name}_to_channels_last")(x)
    x = LayerNorm(dim, name=f"{name}_norm")(x)
    x = Dense(4 * dim, name=f"{name}_pwconv1")(x)
    x = Activation("gelu", name=f"{name}_act")(x)
    x = Dense(dim, name=f"{name}_pwconv2")(x)
    if layer_scale_init_value > 0:
        x = LayerScale(dim, layer_scale_init_value, f"{name}_gamma")(x)
    x = Permute((3, 1, 2), name=f"{name}_to_channels_first")(x)
    x = drop_samples(x, drop_path, f"{name}_drop_path")
    return Add(name=f"{name}_add")([shortcut, x])


def drop_samples(x, drop_path, name):
    if drop_path > 0.0:
        x = Dropout(drop_path, noise_shape=(None, 1, 1, 1), name=name)(x)
    return x


def pool_and_normalize(x, dim):
    pooled = GlobalAveragePooling2D("channels_first", name="pool")(x)
    pooled = Reshape((1, dim), name="pool_token")(pooled)
    patches = Reshape((dim, -1), name="flatten")(x)
    patches = Permute((2, 1), name="prenorm")(patches)
    tokens = Concatenate(axis=1, name="tokens")([pooled, patches])
    normalized = LayerNormalization(epsilon=1e-6, name="norm")(tokens)
    return normalized[:, 0], normalized[:, 1:]


convnext_sizes = {
    "tiny": dict(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]),
    "small": dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768]),
    "base": dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]),
    "large": dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536]),
}


def get_convnext_arch(arch_name):
    query_sizename = arch_name.split("_")[1]
    try:
        size_dict = convnext_sizes[query_sizename]
    except KeyError:
        raise NotImplementedError("didn't recognize vit size string")
    return partial(ConvNeXt, **size_dict)
