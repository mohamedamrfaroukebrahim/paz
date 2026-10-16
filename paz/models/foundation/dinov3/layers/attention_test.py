import os
import pytest
import torch
import numpy as np
from paz.models.foundation.dinov3.layers.attention import (
    SelfAttention,
    CausalSelfAttention,
)
from paz.models.foundation.dinov3.layers.torch_layers_for_testing import (
    PT_SelfAttention,
    PT_CausalSelfAttention,
)


def port_dense(torch_layer, keras_layer):
    weights = [torch_layer.weight.T.detach().numpy()]
    if torch_layer.bias is not None:
        weights.append(torch_layer.bias.detach().numpy())
    keras_layer.set_weights(weights)


def port_weights(torch_model, keras_model):
    """Copies weights from a PyTorch model to its Keras equivalent."""
    qkv = keras_model.get_layer(f"{keras_model.name}_qkv")
    proj = keras_model.get_layer(f"{keras_model.name}_proj")
    port_dense(torch_model.qkv, qkv)
    port_dense(torch_model.proj, proj)


@pytest.fixture(scope="module")
def params():
    """Provides common parameters for the tests."""
    torch.manual_seed(0)
    np.random.seed(0)
    return {"DIM": 128, "NUM_HEADS": 4, "BATCH": 2, "SEQ_LEN": 64}


@pytest.fixture(scope="module")
def inputs(params):
    """Provides random input tensors for Torch and Keras."""
    p = params
    x_torch = torch.rand(p["BATCH"], p["SEQ_LEN"], p["DIM"])
    shape = (1, p["NUM_HEADS"], p["SEQ_LEN"], p["DIM"] // p["NUM_HEADS"])
    rope_torch = (torch.rand(*shape), torch.rand(*shape))
    x_keras = x_torch.numpy()
    rope_keras = (rope_torch[0].numpy(), rope_torch[1].numpy())
    return x_torch, x_keras, rope_torch, rope_keras


@pytest.fixture(scope="module")
def self_attention_models(params):
    """Provides initialized and weight-ported SelfAttention models."""
    p = params
    bias_kwargs = dict(qkv_bias=True, proj_bias=True)
    torch_attn = PT_SelfAttention(p["DIM"], p["NUM_HEADS"], **bias_kwargs)
    kwargs = dict(qkv_bias=True, proj_bias=True, use_rope=True)
    keras_attn = SelfAttention(p["DIM"], p["NUM_HEADS"], **kwargs)
    torch_attn.eval()

    port_weights(torch_attn, keras_attn)
    return torch_attn, keras_attn


@pytest.fixture(scope="module")
def causal_self_attention_models(params):
    """Provides initialized and weight-ported CausalSelfAttention models."""
    p = params
    args = (p["DIM"], p["NUM_HEADS"])
    bias_kwargs = dict(qkv_bias=True, proj_bias=True)
    torch_causal_attn = PT_CausalSelfAttention(*args, **bias_kwargs)
    keras_causal_attn = CausalSelfAttention(*args, **bias_kwargs)
    torch_causal_attn.eval()

    shape = (p["BATCH"], p["SEQ_LEN"], p["DIM"])
    _ = keras_causal_attn(np.zeros(shape, dtype="float32"))
    port_weights(torch_causal_attn, keras_causal_attn)
    return torch_causal_attn, keras_causal_attn


def test_self_attention_call(self_attention_models, inputs):
    """Tests the `call` method of SelfAttention with RoPE."""
    torch_attn, keras_attn = self_attention_models
    x_torch, x_keras, rope_torch, rope_keras = inputs

    torch_out = torch_attn(x_torch, rope=rope_torch).detach().numpy()
    keras_out = np.array(keras_attn([x_keras, *rope_keras], training=False))

    message = "`call` method outputs differ."
    np.testing.assert_allclose(torch_out, keras_out, atol=1e-5, err_msg=message)


def test_self_attention_forward_list(self_attention_models, inputs):
    """Tests the `forward_list` method of SelfAttention."""
    torch_attn, keras_attn = self_attention_models
    x_torch, x_keras, rope_torch, rope_keras = inputs

    kwargs = dict(x_list=[x_torch, x_torch], rope_list=[rope_torch, rope_torch])
    torch_out_list = torch_attn.forward_list(**kwargs)
    x_list, rope_list = [x_keras, x_keras], [rope_keras, rope_keras]
    keras_out_list = [
        keras_attn([x, *rope], training=False)
        for x, rope in zip(x_list, rope_list)
    ]

    torch_out = torch_out_list[0].detach().numpy()
    keras_out = np.array(keras_out_list[0])
    message = "`forward_list` method outputs differ."
    np.testing.assert_allclose(torch_out, keras_out, atol=1e-5, err_msg=message)


def test_causal_self_attention_call(causal_self_attention_models, inputs):
    """Tests the `call` method of CausalSelfAttention."""
    torch_causal_attn, keras_causal_attn = causal_self_attention_models
    x_torch, x_keras, _, _ = inputs

    torch_out = torch_causal_attn(x_torch).detach().numpy()
    keras_out = np.array(keras_causal_attn(x_keras, training=False))

    message = "Causal `call` method outputs differ."
    np.testing.assert_allclose(torch_out, keras_out, atol=1e-5, err_msg=message)


def test_self_attention_without_rope(self_attention_models, inputs):
    """Tests the SelfAttention layer without applying RoPE."""
    torch_attn, keras_attn = self_attention_models
    x_torch, x_keras, _, _ = inputs
    dim, num_heads = x_keras.shape[-1], torch_attn.num_heads
    kwargs = dict(qkv_bias=True, proj_bias=True, use_rope=False)
    keras_attn = SelfAttention(dim, num_heads, **kwargs)
    port_weights(torch_attn, keras_attn)

    torch_out = torch_attn(x_torch, rope=None).detach().numpy()
    keras_out = np.array(keras_attn(x_keras, training=False))

    message = "`call` method without RoPE failed."
    np.testing.assert_allclose(torch_out, keras_out, atol=1e-5, err_msg=message)


def test_causal_attention_as_standard_attention(
        causal_self_attention_models, inputs):
    """Tests CausalSelfAttention with `is_causal=False` to ensure it matches
    standard attention."""
    torch_causal_attn, keras_causal_attn = causal_self_attention_models
    x_torch, x_keras, _, _ = inputs
    dim, num_heads = torch_causal_attn.dim, torch_causal_attn.num_heads
    kwargs = dict(qkv_bias=True, proj_bias=True, is_causal=False)
    keras_causal_attn = CausalSelfAttention(dim, num_heads, **kwargs)
    port_weights(torch_causal_attn, keras_causal_attn)

    torch_out = torch_causal_attn(x_torch, is_causal=False).detach().numpy()
    keras_out = np.array(keras_causal_attn(x_keras, training=False))

    message = "Causal layer with `is_causal=False` failed."
    np.testing.assert_allclose(torch_out, keras_out, atol=1e-5, err_msg=message)


def test_dropout_activation(params, inputs):
    """Tests that dropout layers are active during training and inactive
    during evaluation."""
    p = params
    _, x_keras, _, _ = inputs
    kwargs = dict(attn_drop=0.5, proj_drop=0.5)
    keras_attn_dropout = SelfAttention(p["DIM"], p["NUM_HEADS"], **kwargs)
    _ = keras_attn_dropout(x_keras)

    out_train = keras_attn_dropout(x_keras, training=True)
    out_eval = keras_attn_dropout(x_keras, training=False)

    message = "Dropout layers seem inactive during training."
    assert not np.allclose(out_train, out_eval), message


def test_forward_list_varied_shapes(self_attention_models, params, inputs):
    """Tests `forward_list` with a list of tensors having different sequence
    lengths."""
    torch_attn, keras_attn = self_attention_models
    p = params
    x_torch, x_keras, rope_torch, rope_keras = inputs

    x_torch_short = torch.rand(p["BATCH"], p["SEQ_LEN"] // 2, p["DIM"])
    x_keras_short = x_torch_short.numpy()
    shape = (1, p["NUM_HEADS"], p["SEQ_LEN"] // 2, p["DIM"] // p["NUM_HEADS"])
    rope_torch_short = (torch.rand(*shape), torch.rand(*shape))
    rope_keras_short = tuple(rope.numpy() for rope in rope_torch_short)

    kwargs = dict(x_list=[x_torch, x_torch_short])
    kwargs.update(rope_list=[rope_torch, rope_torch_short])
    torch_list_varied = torch_attn.forward_list(**kwargs)
    x_list = [x_keras, x_keras_short]
    rope_list = [rope_keras, rope_keras_short]
    keras_list_varied = [
        keras_attn([x, *rope], training=False)
        for x, rope in zip(x_list, rope_list)
    ]

    torch_out = torch_list_varied[0].detach().numpy()
    np.testing.assert_allclose(torch_out, keras_list_varied[0], atol=1e-5)
    torch_out = torch_list_varied[1].detach().numpy()
    np.testing.assert_allclose(torch_out, keras_list_varied[1], atol=1e-5)


def test_initialization_no_biases(params, inputs):
    """Tests model initialization and execution with biases set to False."""
    p = params
    x_torch, x_keras, rope_torch, rope_keras = inputs
    bias_kwargs = dict(qkv_bias=False, proj_bias=False)
    args = (p["DIM"], p["NUM_HEADS"])
    torch_attn_no_bias = PT_SelfAttention(*args, **bias_kwargs)
    kwargs = dict(qkv_bias=False, proj_bias=False, use_rope=True)
    keras_attn_no_bias = SelfAttention(p["DIM"], p["NUM_HEADS"], **kwargs)
    torch_attn_no_bias.eval()

    port_weights(torch_attn_no_bias, keras_attn_no_bias)

    torch_out = torch_attn_no_bias(x_torch, rope=rope_torch).detach().numpy()
    keras_inputs = [x_keras, *rope_keras]
    keras_out = np.array(keras_attn_no_bias(keras_inputs, training=False))

    message = "Model with no biases failed."
    np.testing.assert_allclose(torch_out, keras_out, atol=1e-5, err_msg=message)


def test_value_error_on_invalid_heads(params):
    """Tests that a ValueError is raised if `dim` is not divisible by
    `num_heads`."""
    with pytest.raises(ValueError, match="must be divisible by"):
        SelfAttention(dim=params["DIM"], num_heads=7)


DINO_WEIGHT_PATH = "/path/that/does/not/exist/dinov3_vits16_pretrain.pth"
dinov3_files_exist = os.path.isfile(DINO_WEIGHT_PATH)
SKIP_REASON = "DINOv3 local model/weight files not found."


@pytest.mark.skipif(not dinov3_files_exist, reason=SKIP_REASON)
def test_dinov3_pretrained_weights_match():
    """
    Tests the Keras SelfAttention layer by loading weights from a pre-trained
    DINOv3 PyTorch model and verifying the outputs match.
    """
    from paz.models.foundation.dinov3.models.torch_vision_transformer_for_testing import (
        PT_vit_small,
    )

    model_kwargs = {
        "img_size": 224,
        "patch_size": 16,
        "ffn_layer": "mlp",
        "untie_cls_and_patch_norms": False,
        "norm_layer": "layernorm",
        "layerscale_init": 1e-6,
        "n_storage_tokens": 4,
        "pos_embed_rope_dtype": "float32",
    }
    dinov3_model_local = PT_vit_small(**model_kwargs)
    state_dict = torch.load(DINO_WEIGHT_PATH, map_location=torch.device("cpu"))
    dinov3_model_local.load_state_dict(state_dict, strict=False)
    dinov3_model_local.eval()

    torch_pretrained_attn = dinov3_model_local.blocks[0].attn
    PRETRAINED_DIM = dinov3_model_local.embed_dim
    PRETRAINED_NUM_HEADS = torch_pretrained_attn.num_heads
    QKV_BIAS = torch_pretrained_attn.qkv.bias is not None
    PROJ_BIAS = torch_pretrained_attn.proj.bias is not None

    kwargs = dict(dim=PRETRAINED_DIM, num_heads=PRETRAINED_NUM_HEADS)
    kwargs.update(qkv_bias=QKV_BIAS, proj_bias=PROJ_BIAS)
    keras_attn_pretrained = SelfAttention(**kwargs)

    BATCH_SIZE = 2
    NUM_TOKENS = 201
    x_torch_pretrained = torch.rand(BATCH_SIZE, NUM_TOKENS, PRETRAINED_DIM)
    x_keras_pretrained = x_torch_pretrained.numpy()

    zeros = np.zeros_like(x_keras_pretrained, dtype="float32")
    _ = keras_attn_pretrained(zeros)
    port_weights(torch_pretrained_attn, keras_attn_pretrained)

    torch_out_pretrained = torch_pretrained_attn(x_torch_pretrained)
    torch_out_pretrained = torch_out_pretrained.detach().numpy()
    outputs = keras_attn_pretrained(x_keras_pretrained, training=False)
    keras_out_pretrained = np.array(outputs)

    message = "Keras output with DINOv3 weights does not match PyTorch output."
    args = (torch_out_pretrained, keras_out_pretrained)
    np.testing.assert_allclose(*args, atol=1e-5, err_msg=message)
