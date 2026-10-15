import keras
from keras.layers import EinsumDense


def LayerScale(dimension, init_values=1e-5, name=None):
    """Per-channel learnable scale, held as the kernel of an EinsumDense."""
    initializer = keras.initializers.Constant(init_values)
    kwargs = dict(output_shape=(dimension,), bias_axes=None, name=name)
    kwargs["kernel_initializer"] = initializer
    return EinsumDense("...d,d->...d", **kwargs)
