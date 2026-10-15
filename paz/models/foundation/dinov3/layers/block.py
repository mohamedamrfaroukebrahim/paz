from keras import Input, Model
from keras.layers import Add, Dropout, LayerNormalization

from paz.layers import RMSNormalization
from paz.models.foundation.dinov3.layers.attention import CausalSelfAttention
from paz.models.foundation.dinov3.layers.attention import SelfAttention
from paz.models.foundation.dinov3.layers.attention import build_rope_inputs
from paz.models.foundation.dinov3.layers.ffn_layers import Mlp
from paz.models.foundation.dinov3.layers.ffn_layers import build_ffn
from paz.models.foundation.dinov3.layers.layer_scale import LayerScale


def SelfAttentionBlock(dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                       proj_bias=True, ffn_bias=True, drop=0.0, attn_drop=0.0,
                       drop_path=0.0, init_values=None, act_layer="gelu",
                       norm_layer="layernorm", ffn_layer="mlp", use_rope=False,
                       num_prefix_tokens=0, name="block"):
    tokens = Input((None, dim), name=f"{name}_tokens")
    rope = build_rope_inputs(dim // num_heads, name) if use_rope else None
    args = (dim, num_heads, mlp_ratio, qkv_bias, proj_bias, ffn_bias, drop)
    args = args + (attn_drop, drop_path, init_values, act_layer, norm_layer)
    args = args + (ffn_layer, num_prefix_tokens, name)
    outputs = apply_self_attention_block(tokens, rope, *args)
    inputs = tokens if rope is None else [tokens, *rope]
    return Model(inputs, outputs, name=name)


def CausalSelfAttentionBlock(dim, num_heads, mlp_ratio=4.0, init_values=None,
                             is_causal=True, act_layer="gelu",
                             norm_layer="layernorm", drop=0.0, qkv_bias=True,
                             name="block"):
    tokens = Input((None, dim), name=f"{name}_tokens")
    args = (dim, num_heads, qkv_bias, True, drop, drop, is_causal)
    attention = CausalSelfAttention(*args, f"{name}_attention")
    normed = build_norm(norm_layer, f"{name}_attention_norm")(tokens)
    x = add_residual(tokens, attention(normed), dim, init_values, 0.0, name, 1)
    args = (dim, int(dim * mlp_ratio), dim, act_layer, drop, True, False)
    feed_forward = Mlp(*args, f"{name}_feed_forward")
    normed = build_norm(norm_layer, f"{name}_ffn_norm")(x)
    args = (dim, init_values, 0.0, name, 2)
    outputs = add_residual(x, feed_forward(normed), *args)
    return Model(tokens, outputs, name=name)


def apply_self_attention_block(x, rope, dim, num_heads, mlp_ratio, qkv_bias,
                               proj_bias, ffn_bias, drop, attn_drop,
                               drop_path, init_values, act_layer, norm_layer,
                               ffn_layer, num_prefix_tokens, name):
    """Pre-norm block: attention then feedforward, each with a residual."""
    args = (dim, num_heads, qkv_bias, proj_bias, attn_drop, drop)
    args = args + (rope is not None, num_prefix_tokens, f"{name}_attn")
    attention = SelfAttention(*args)
    normed = build_norm(norm_layer, f"{name}_norm1")(x)
    attended = attention(normed if rope is None else [normed, *rope])
    x = add_residual(x, attended, dim, init_values, drop_path, name, 1)
    hidden_features = int(dim * mlp_ratio)
    args = (ffn_layer, dim, hidden_features, act_layer, drop, ffn_bias)
    feedforward = build_ffn(*args, f"{name}_mlp")
    normed = build_norm(norm_layer, f"{name}_norm2")(x)
    args = (dim, init_values, drop_path, name, 2)
    return add_residual(x, feedforward(normed), *args)


def add_residual(x, branch, dim, init_values, drop_path, name, index):
    if init_values is not None:
        branch = LayerScale(dim, init_values, f"{name}_ls{index}")(branch)
    branch = drop_samples(branch, drop_path, f"{name}_drop_path{index}")
    return Add(name=f"{name}_add{index}")([x, branch])


def drop_samples(x, drop_path, name):
    if drop_path > 0.0:
        x = Dropout(drop_path, noise_shape=(None, 1, 1), name=name)(x)
    return x


def build_norm(norm_layer, name):
    if norm_layer == "layernorm":
        return LayerNormalization(epsilon=1e-6, name=name)
    if norm_layer == "rmsnorm":
        return RMSNormalization(epsilon=1e-6, name=name)
    raise ValueError(f"Unknown norm_layer: {norm_layer}")
