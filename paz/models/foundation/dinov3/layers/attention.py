from keras import Input, Model, ops
from keras.layers import Dense, Dropout, Reshape

from paz.models.transformers.embeddings.rotary import rotate_half


def SelfAttention(dim, num_heads=8, qkv_bias=False, proj_bias=True,
                  attn_drop=0.0, proj_drop=0.0, use_rope=False,
                  num_prefix_tokens=0, name="attn"):
    """Multi-head self attention returned as a ``keras.Model``.

    With ``use_rope`` the model takes ``[tokens, sin, cos]`` and rotates
    every query and key except the first ``num_prefix_tokens``.
    """
    head_dim = compute_head_dim(dim, num_heads)
    tokens = Input((None, dim), name=f"{name}_tokens")
    rope = build_rope_inputs(head_dim, name) if use_rope else None
    qkv = project_qkv(tokens, dim, qkv_bias, name)
    args = (qkv, num_heads, head_dim, rope, num_prefix_tokens, attn_drop)
    context = attend(*args, False, name)
    outputs = project_output(context, dim, proj_bias, proj_drop, name)
    inputs = tokens if rope is None else [tokens, *rope]
    return Model(inputs, outputs, name=name)


def CausalSelfAttention(dim, num_heads=8, qkv_bias=False, proj_bias=True,
                        attn_drop=0.0, proj_drop=0.0, is_causal=True,
                        name="attention"):
    head_dim = compute_head_dim(dim, num_heads)
    tokens = Input((None, dim), name=f"{name}_tokens")
    qkv = project_qkv(tokens, dim, qkv_bias, name)
    args = (qkv, num_heads, head_dim, None, 0, attn_drop, is_causal)
    context = attend(*args, name)
    outputs = project_output(context, dim, proj_bias, proj_drop, name)
    return Model(tokens, outputs, name=name)


def compute_head_dim(dim, num_heads):
    if dim % num_heads != 0:
        message = f"dim ({dim}) must be divisible by num_heads ({num_heads})."
        raise ValueError(message)
    return dim // num_heads


def build_rope_inputs(head_dim, name):
    sin = Input((None, None, head_dim), name=f"{name}_sin")
    cos = Input((None, None, head_dim), name=f"{name}_cos")
    return sin, cos


def project_qkv(tokens, dim, qkv_bias, name):
    return Dense(dim * 3, use_bias=qkv_bias, name=f"{name}_qkv")(tokens)


def project_output(context, dim, proj_bias, proj_drop, name):
    outputs = Dense(dim, use_bias=proj_bias, name=f"{name}_proj")(context)
    return apply_dropout(outputs, proj_drop, f"{name}_proj_drop")


def attend(qkv, num_heads, head_dim, rope, num_prefix_tokens, attn_drop,
           is_causal, name):
    query, key, value = split_heads(qkv, num_heads, head_dim, name)
    if rope is not None:
        query, key = apply_rope(query, key, rope, num_prefix_tokens)
    scores = compute_scores(query, key, head_dim)
    if is_causal:
        scores = scores + build_causal_mask(scores)
    weights = ops.softmax(scores, axis=-1)
    weights = apply_dropout(weights, attn_drop, f"{name}_attn_drop")
    context = ops.matmul(weights, value)
    return merge_heads(context, num_heads * head_dim, name)


def split_heads(qkv, num_heads, head_dim, name):
    shape = (-1, 3, num_heads, head_dim)
    heads = Reshape(shape, name=f"{name}_split_heads")(qkv)
    heads = ops.transpose(heads, (2, 0, 3, 1, 4))
    return heads[0], heads[1], heads[2]


def merge_heads(context, dim, name):
    context = ops.transpose(context, (0, 2, 1, 3))
    return Reshape((-1, dim), name=f"{name}_merge_heads")(context)


def compute_scores(query, key, head_dim):
    scores = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2)))
    return scores * head_dim**-0.5


def build_causal_mask(scores):
    ones = ops.ones_like(scores[0, 0])
    return ops.triu(ones, k=1) * -1e9


def apply_rope(query, key, rope, num_prefix_tokens):
    sin, cos = rope
    query = rotate_tokens(query, sin, cos, num_prefix_tokens)
    key = rotate_tokens(key, sin, cos, num_prefix_tokens)
    return query, key


def rotate_tokens(x, sin, cos, num_prefix_tokens):
    if num_prefix_tokens == 0:
        return apply_rotary_position_embedding(x, sin, cos)
    prefix = x[:, :, :num_prefix_tokens, :]
    rotated = x[:, :, num_prefix_tokens:, :]
    rotated = apply_rotary_position_embedding(rotated, sin, cos)
    return ops.concatenate((prefix, rotated), axis=-2)


def apply_rotary_position_embedding(x, sin, cos):
    return (x * cos) + (rotate_half(x) * sin)


def apply_dropout(x, rate, name):
    if rate > 0.0:
        x = Dropout(rate, name=name)(x)
    return x
