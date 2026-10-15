from keras import Input, Model, activations
from keras.layers import Dense

from paz.models.foundation.dinov3.layers.attention import apply_dropout

SWIGLU_ALIGNMENTS = dict(swiglu=8, swiglu32=32, swiglu64=64, swiglu128=128)


def Mlp(in_features, hidden_features=None, out_features=None,
        act_layer="gelu", drop=0.0, bias=True, approximate_gelu=False,
        name="mlp"):
    hidden_features = hidden_features or in_features
    out_features = out_features or in_features
    tokens = Input((None, in_features), name=f"{name}_tokens")
    args = (tokens, hidden_features, out_features, act_layer, drop, bias)
    outputs = apply_mlp(*args, approximate_gelu, name)
    return Model(tokens, outputs, name=name)


def SwiGLUFFN(in_features, hidden_features=None, out_features=None,
              bias=True, align_to=8, name="mlp"):
    hidden_features = hidden_features or in_features
    out_features = out_features or in_features
    tokens = Input((None, in_features), name=f"{name}_tokens")
    args = (tokens, hidden_features, out_features, bias, align_to, name)
    outputs = apply_swiglu(*args)
    return Model(tokens, outputs, name=name)


def build_ffn(ffn_layer, dim, hidden_features, act_layer, drop, bias, name):
    if ffn_layer == "mlp":
        args = (dim, hidden_features, dim, act_layer, drop, bias, False)
        return Mlp(*args, name)
    if ffn_layer in SWIGLU_ALIGNMENTS:
        align_to = SWIGLU_ALIGNMENTS[ffn_layer]
        return SwiGLUFFN(dim, hidden_features, dim, bias, align_to, name)
    raise ValueError(f"Unknown ffn_layer: {ffn_layer}")


def apply_mlp(x, hidden_features, out_features, act_layer, drop, bias,
              approximate_gelu, name):
    x = Dense(hidden_features, use_bias=bias, name=f"{name}_fc1")(x)
    x = activate(x, act_layer, approximate_gelu)
    x = apply_dropout(x, drop, f"{name}_drop1")
    x = Dense(out_features, use_bias=bias, name=f"{name}_fc2")(x)
    return apply_dropout(x, drop, f"{name}_drop2")


def apply_swiglu(x, hidden_features, out_features, bias, align_to, name):
    inner_features = align_features(hidden_features, align_to)
    gate = Dense(inner_features, use_bias=bias, name=f"{name}_w1")(x)
    up = Dense(inner_features, use_bias=bias, name=f"{name}_w2")(x)
    hidden = activations.silu(gate) * up
    return Dense(out_features, use_bias=bias, name=f"{name}_w3")(hidden)


def align_features(hidden_features, align_to):
    features = int(hidden_features * 2 / 3)
    return (features + align_to - 1) // align_to * align_to


def activate(x, act_layer, approximate_gelu):
    if act_layer == "gelu":
        return activations.gelu(x, approximate=approximate_gelu)
    return activations.get(act_layer)(x)
