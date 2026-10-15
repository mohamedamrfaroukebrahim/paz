import math
from functools import partial

from keras import ops, random
from keras.layers import EinsumDense, Reshape


def RopePositionEmbedding(embed_dim, num_heads, base=100.0, min_period=None,
                          max_period=None, normalize_coords="separate",
                          shift_coords=None, jitter_coords=None,
                          rescale_coords=None, dtype="float32", periods=None):
    """Returns ``rope(H, W, training=False) -> (sin, cos)``.

    ``periods`` overrides the periods computed from ``base`` or from
    ``min_period`` and ``max_period``, e.g. with ported weights.
    """
    validate_rope(embed_dim, num_heads, base, min_period, max_period)
    if periods is None:
        args = (embed_dim // num_heads, base, min_period, max_period, dtype)
        periods = compute_periods(*args)
    kwargs = dict(periods=periods, normalize_coords=normalize_coords)
    kwargs.update(shift_coords=shift_coords, jitter_coords=jitter_coords)
    kwargs.update(rescale_coords=rescale_coords, dtype=dtype)
    return partial(compute_rope, **kwargs)


def apply_rope_embed(reference, height, width, head_dim, normalize_coords,
                     name):
    """Builds inference RoPE tables ``(1, 1, H * W, head_dim)`` in a graph.

    The periods live in the non-trainable kernel of the ``name`` layer,
    which scales a tensor of ones drawn from ``reference``.
    """
    num_periods = head_dim // 4
    ones = ops.ones_like(reference[:1, :1, :num_periods])
    periods = build_periods_layer(num_periods, name)(ones)[0, 0]
    sin, cos = compute_rope(height, width, periods, normalize_coords)
    shape = (1, height * width, head_dim)
    sin = Reshape(shape, name=f"{name}_sin")(ops.expand_dims(sin, 0))
    cos = Reshape(shape, name=f"{name}_cos")(ops.expand_dims(cos, 0))
    return sin, cos


def build_periods_layer(num_periods, name):
    kwargs = dict(output_shape=(num_periods,), bias_axes=None, name=name)
    kwargs.update(kernel_initializer="ones", trainable=False)
    return EinsumDense("...d,d->...d", **kwargs)


def validate_rope(embed_dim, num_heads, base, min_period, max_period):
    if embed_dim % (4 * num_heads) != 0:
        message = f"embed_dim ({embed_dim}) must be divisible by "
        raise ValueError(message + f"4 * num_heads ({4 * num_heads}).")
    both_periods = min_period is not None and max_period is not None
    missing = base is None and not both_periods
    duplicated = base is not None and both_periods
    if missing or duplicated:
        message = "Either `base` or `min_period`+`max_period` must be "
        raise ValueError(message + "provided.")


def compute_periods(head_dim, base, min_period, max_period, dtype="float32"):
    num_periods = head_dim // 4
    if base is not None:
        exponents = 2 * ops.arange(num_periods, dtype=dtype) / (head_dim // 2)
        return ops.power(base, exponents)
    ratio = max_period / min_period
    exponents = ops.linspace(0, 1, num_periods, dtype=dtype)
    periods = ops.power(ratio, exponents)
    periods = periods / ratio
    return periods * max_period


def compute_rope(H, W, periods, normalize_coords="separate", shift_coords=None,
                 jitter_coords=None, rescale_coords=None, training=False,
                 dtype="float32"):
    coords = compute_coords(H, W, normalize_coords, dtype)
    if training:
        args = (shift_coords, jitter_coords, rescale_coords, dtype)
        coords = augment_coords(coords, *args)
    numerators = 2.0 * math.pi * ops.expand_dims(coords, axis=-1)
    periods = ops.expand_dims(ops.expand_dims(periods, axis=0), axis=0)
    angles = ops.divide(numerators, periods)
    angles = ops.reshape(angles, (-1, 2 * periods.shape[-1]))
    angles = ops.tile(angles, [1, 2])
    return ops.sin(angles), ops.cos(angles)


def compute_coords(H, W, normalize_coords, dtype):
    coords_h = ops.cast(ops.arange(H, dtype="int32"), dtype=dtype) + 0.5
    coords_w = ops.cast(ops.arange(W, dtype="int32"), dtype=dtype) + 0.5
    if normalize_coords == "max":
        norm_h, norm_w = max(H, W), max(H, W)
    elif normalize_coords == "min":
        norm_h, norm_w = min(H, W), min(H, W)
    elif normalize_coords == "separate":
        norm_h, norm_w = H, W
    else:
        raise ValueError(f"Unknown normalize_coords: {normalize_coords}")
    coords_h, coords_w = coords_h / norm_h, coords_w / norm_w
    grid_h, grid_w = ops.meshgrid(coords_h, coords_w, indexing="ij")
    coords = ops.reshape(ops.stack([grid_h, grid_w], axis=-1), (-1, 2))
    return 2.0 * coords - 1.0


def augment_coords(coords, shift_coords, jitter_coords, rescale_coords,
                   dtype):
    if shift_coords is not None:
        bounds = (-shift_coords, shift_coords)
        coords = coords + random.uniform((2,), *bounds, dtype=dtype)
    if jitter_coords is not None:
        coords = coords * sample_log_uniform((2,), jitter_coords, dtype)
    if rescale_coords is not None:
        coords = coords * sample_log_uniform((1,), rescale_coords, dtype)
    return coords


def sample_log_uniform(shape, scale, dtype):
    bound = math.log(scale)
    return ops.exp(random.uniform(shape, -bound, bound, dtype=dtype))
