import keras
import numpy as np


def cat_keep_shapes(x_list):
    """Flattens and concatenates tensors, keeping their original shapes."""
    shapes = [x.shape for x in x_list]
    num_tokens = [count_tokens(x) for x in x_list]
    pairs = zip(x_list, num_tokens)
    reshaped_tensors = [flatten_tokens(x, n) for x, n in pairs]
    flattened = keras.ops.concatenate(reshaped_tensors, axis=0)
    return flattened, shapes, num_tokens


def flatten_tokens(x, num_tokens):
    return keras.ops.reshape(x, (num_tokens, x.shape[-1]))


def count_tokens(x):
    if len(x.shape) > 1:
        return np.prod(x.shape[:-1]).item()
    return x.shape[0]


def uncat_with_shapes(flattened, shapes, num_tokens):
    """Reverses the cat_keep_shapes operation."""
    if len(num_tokens) == 1:
        outputs_splitted = [flattened]
    else:
        split_indices = np.cumsum(num_tokens)[:-1].tolist()
        outputs_splitted = keras.ops.split(flattened, split_indices, axis=0)

    new_feature_dim = flattened.shape[-1]
    shapes_adjusted = [shape[:-1] + (new_feature_dim,) for shape in shapes]

    pairs = zip(outputs_splitted, shapes_adjusted)
    outputs_reshaped = [keras.ops.reshape(o, s) for o, s in pairs]
    return outputs_reshaped


def named_apply(fn, module, name="", depth_first=True, include_root=False):
    if not depth_first and include_root:
        fn(module=module, name=name)

    for child_name, child_module in vars(module).items():
        if isinstance(child_module, keras.Layer):
            names = (name, child_name)
            full_child_name = ".".join(names) if name else child_name
            kwargs = dict(fn=fn, module=child_module, name=full_child_name)
            named_apply(**kwargs, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)

    return module
