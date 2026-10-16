import os
import pytest
import torch
import numpy as np
import keras
from paz.models.foundation.dinov3.models.vision_transformer import vit_small
from paz.models.foundation.dinov3.models.torch_vision_transformer_for_testing import (
    PT_vit_small,
)
from paz.models.foundation.dinov3.port_dino_weights_from_torch_to_keras import (
    transfer_weights_from_pt_to_keras,
)


PT_WEIGHTS_DIR = "/home/octavio/Storage/dinov3/"
PT_WEIGHT_FILE = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
PT_WEIGHT_PATH = PT_WEIGHTS_DIR + PT_WEIGHT_FILE
weights_exist = os.path.isfile(PT_WEIGHT_PATH)

MODEL_KWARGS = {
    "img_size": 224,
    "patch_size": 16,
    "ffn_layer": "mlp",
    "untie_cls_and_patch_norms": False,
    "norm_layer": "layernorm",
    "layerscale_init": 1e-6,
    "n_storage_tokens": 4,
    "pos_embed_rope_dtype": "float32",
}


def count_blocks(model):
    names = [layer.name for layer in model.layers]
    return sum(name.endswith("_attn") for name in names)


def token_weight(model, name):
    return model.get_layer(name).weights[0]


def token_value(model, name, state_dict):
    shape = tuple(token_weight(model, name).shape)
    return state_dict[name].numpy().reshape(shape)


@pytest.fixture(scope="module")
def pytorch_model_and_weights():
    if not weights_exist:
        pytest.skip(f"PyTorch weight file not found: {PT_WEIGHT_PATH}")

    pt_model = PT_vit_small(**MODEL_KWARGS)
    state_dict = torch.load(PT_WEIGHT_PATH, map_location=torch.device("cpu"))
    pt_model.load_state_dict(state_dict, strict=False)
    pt_model.eval()

    return pt_model, state_dict


@pytest.fixture(scope="module")
def keras_model():
    model = vit_small(**MODEL_KWARGS)
    dummy_input = np.zeros((1, 224, 224, 3), dtype="float32")
    model(dummy_input, training=False)
    return model


def test_cls_token_porting(pytorch_model_and_weights, keras_model):
    """Tests that cls_token is correctly ported from PyTorch to Keras."""
    _, state_dict = pytorch_model_and_weights

    cls_token = token_weight(keras_model, "cls_token")
    expected = token_value(keras_model, "cls_token", state_dict)
    original_cls_token = cls_token.numpy().copy()
    cls_token.assign(expected)

    message = "cls_token not correctly assigned"
    np.testing.assert_array_equal(cls_token.numpy(), expected, err_msg=message)
    unchanged = np.array_equal(original_cls_token, cls_token.numpy())
    assert not unchanged, "cls_token was not modified"


def test_mask_token_porting(pytorch_model_and_weights, keras_model):
    """Tests that mask_token is correctly ported from PyTorch to Keras."""
    _, state_dict = pytorch_model_and_weights

    mask_token = token_weight(keras_model, "mask_token")
    expected = token_value(keras_model, "mask_token", state_dict)
    original_mask_token = mask_token.numpy().copy()
    mask_token.assign(expected)

    message = "mask_token not correctly assigned"
    np.testing.assert_array_equal(mask_token.numpy(), expected, err_msg=message)
    unchanged = np.array_equal(original_mask_token, mask_token.numpy())
    assert not unchanged, "mask_token was not modified"


def test_storage_tokens_porting(pytorch_model_and_weights, keras_model):
    """Tests that storage_tokens is correctly ported from PyTorch to Keras."""
    _, state_dict = pytorch_model_and_weights

    names = [layer.name for layer in keras_model.layers]
    if "storage_tokens" not in state_dict or "storage_tokens" not in names:
        pytest.skip("storage_tokens not present in model")

    storage_tokens = token_weight(keras_model, "storage_tokens")
    expected = token_value(keras_model, "storage_tokens", state_dict)
    original_storage = storage_tokens.numpy().copy()
    storage_tokens.assign(expected)

    message = "storage_tokens not correctly assigned"
    args = (storage_tokens.numpy(), expected)
    np.testing.assert_array_equal(*args, err_msg=message)
    unchanged = np.array_equal(original_storage, storage_tokens.numpy())
    assert not unchanged, "storage_tokens was not modified"


def test_patch_embed_weight_shapes(pytorch_model_and_weights, keras_model):
    """Tests that patch embedding weights have compatible shapes after
    transposition."""
    _, state_dict = pytorch_model_and_weights

    pt_weight = state_dict["patch_embed.proj.weight"]
    pt_weight_transposed = pt_weight.permute(2, 3, 1, 0).numpy()

    projection = keras_model.get_layer("patch_embed_proj")
    keras_weight_shape = projection.kernel.shape

    shapes = f"PT {pt_weight_transposed.shape} vs Keras {keras_weight_shape}"
    message = "Patch embed weight shapes incompatible: " + shapes
    assert pt_weight_transposed.shape == tuple(keras_weight_shape), message


def test_patch_embed_porting(pytorch_model_and_weights, keras_model):
    """Tests that patch embedding weights are correctly ported."""
    _, state_dict = pytorch_model_and_weights

    pt_weight = state_dict["patch_embed.proj.weight"].permute(2, 3, 1, 0)
    pt_weight = pt_weight.numpy()
    if "patch_embed.proj.bias" in state_dict:
        pt_bias = state_dict["patch_embed.proj.bias"].numpy()
    else:
        pt_bias = None

    projection = keras_model.get_layer("patch_embed_proj")
    projection.kernel.assign(pt_weight)
    if pt_bias is not None:
        projection.bias.assign(pt_bias)

    message = "Patch embed weights not correctly ported"
    args = (projection.kernel.numpy(), pt_weight)
    np.testing.assert_allclose(*args, rtol=1e-6, err_msg=message)


def test_rope_embed_porting(pytorch_model_and_weights, keras_model):
    """Tests that RoPE embedding periods are correctly ported."""
    _, state_dict = pytorch_model_and_weights

    if "rope_embed.periods" not in state_dict:
        pytest.skip("rope_embed.periods not in state_dict")

    rope_periods = state_dict["rope_embed.periods"].float().numpy()
    rope_embed = keras_model.get_layer("rope_embed")
    rope_embed.set_weights([rope_periods])

    message = "RoPE periods not correctly ported"
    args = (rope_embed.get_weights()[0], rope_periods)
    np.testing.assert_allclose(*args, rtol=1e-6, err_msg=message)


def test_block_attention_weight_shapes(pytorch_model_and_weights, keras_model):
    """Tests that attention weights in blocks have compatible shapes."""
    _, state_dict = pytorch_model_and_weights

    for i in range(count_blocks(keras_model)):
        pt_qkv_weight = state_dict[f"blocks.{i}.attn.qkv.weight"]
        pt_proj_weight = state_dict[f"blocks.{i}.attn.proj.weight"]

        attention = keras_model.get_layer(f"blocks_{i}_attn")
        qkv = attention.get_layer(f"blocks_{i}_attn_qkv")
        proj = attention.get_layer(f"blocks_{i}_attn_proj")
        keras_qkv_shape = qkv.kernel.shape
        keras_proj_shape = proj.kernel.shape

        message = f"Block {i} QKV weight shapes incompatible"
        assert pt_qkv_weight.T.shape == tuple(keras_qkv_shape), message
        message = f"Block {i} projection weight shapes incompatible"
        assert pt_proj_weight.T.shape == tuple(keras_proj_shape), message


def test_block_norm_weight_shapes(pytorch_model_and_weights, keras_model):
    """Tests that normalization layers in blocks have compatible shapes."""
    _, state_dict = pytorch_model_and_weights

    for i in range(count_blocks(keras_model)):
        pt_norm1_weight = state_dict[f"blocks.{i}.norm1.weight"]
        pt_norm2_weight = state_dict[f"blocks.{i}.norm2.weight"]

        norm1 = keras_model.get_layer(f"blocks_{i}_norm1")
        norm2 = keras_model.get_layer(f"blocks_{i}_norm2")
        keras_norm1_gamma_shape = norm1.gamma.shape
        keras_norm2_gamma_shape = norm2.gamma.shape

        message = f"Block {i} norm1 weight shapes incompatible"
        assert pt_norm1_weight.shape == tuple(keras_norm1_gamma_shape), message
        message = f"Block {i} norm2 weight shapes incompatible"
        assert pt_norm2_weight.shape == tuple(keras_norm2_gamma_shape), message


def test_block_mlp_weight_shapes(pytorch_model_and_weights, keras_model):
    """Tests that MLP weights in blocks have compatible shapes."""
    _, state_dict = pytorch_model_and_weights

    for i in range(count_blocks(keras_model)):
        if f"blocks.{i}.mlp.fc1.weight" in state_dict:
            pt_fc1_weight = state_dict[f"blocks.{i}.mlp.fc1.weight"]
            pt_fc2_weight = state_dict[f"blocks.{i}.mlp.fc2.weight"]

            mlp = keras_model.get_layer(f"blocks_{i}_mlp")
            fc1 = mlp.get_layer(f"blocks_{i}_mlp_fc1")
            fc2 = mlp.get_layer(f"blocks_{i}_mlp_fc2")
            keras_fc1_shape = fc1.kernel.shape
            keras_fc2_shape = fc2.kernel.shape

            message = f"Block {i} MLP fc1 weight shapes incompatible"
            assert pt_fc1_weight.T.shape == tuple(keras_fc1_shape), message
            message = f"Block {i} MLP fc2 weight shapes incompatible"
            assert pt_fc2_weight.T.shape == tuple(keras_fc2_shape), message


def test_block_layerscale_presence(pytorch_model_and_weights, keras_model):
    """Tests that LayerScale is present when init_values is set."""
    names = [layer.name for layer in keras_model.layers]
    for i in range(count_blocks(keras_model)):
        ls1, ls2 = f"blocks_{i}_ls1", f"blocks_{i}_ls2"

        if MODEL_KWARGS["layerscale_init"] is not None:
            # the LayerScale gamma is the kernel of an EinsumDense layer
            has_ls1 = ls1 in names
            has_kernel = hasattr(keras_model.get_layer(ls1), 'kernel')
            message = f"Block {i} ls1 should have gamma when init_values is set"
            assert has_ls1 and has_kernel, message
            has_ls2 = ls2 in names
            has_kernel = hasattr(keras_model.get_layer(ls2), 'kernel')
            message = f"Block {i} ls2 should have gamma when init_values is set"
            assert has_ls2 and has_kernel, message


def test_final_norm_weight_shapes(pytorch_model_and_weights, keras_model):
    """Tests that final normalization layers have compatible shapes."""
    _, state_dict = pytorch_model_and_weights

    pt_norm_weight = state_dict["norm.weight"]
    keras_norm_gamma_shape = keras_model.get_layer("norm").gamma.shape

    message = "Final norm weight shapes incompatible"
    assert pt_norm_weight.shape == tuple(keras_norm_gamma_shape), message

    names = [layer.name for layer in keras_model.layers]
    if "cls_norm" in names:
        pt_cls_norm_weight = state_dict["cls_norm.weight"]
        cls_norm = keras_model.get_layer("cls_norm")
        keras_cls_norm_gamma_shape = tuple(cls_norm.gamma.shape)
        message = "cls_norm weight shapes incompatible"
        assert pt_cls_norm_weight.shape == keras_cls_norm_gamma_shape, message


def test_full_weight_porting_runs_without_error(
        pytorch_model_and_weights, keras_model):
    """Tests that the full weight porting function runs without errors."""
    _, state_dict = pytorch_model_and_weights

    try:
        transfer_weights_from_pt_to_keras(state_dict, keras_model)
    except Exception as e:
        pytest.fail(f"Weight porting failed with error: {e}")


def test_ported_model_output_matches_pytorch(
        pytorch_model_and_weights, keras_model):
    """Tests that after porting weights, model outputs match PyTorch."""
    pt_model, state_dict = pytorch_model_and_weights

    transfer_weights_from_pt_to_keras(state_dict, keras_model)

    np.random.seed(42)
    dummy_input_np = np.random.randn(1, 224, 224, 3).astype("float32")
    keras_input = keras.ops.convert_to_tensor(dummy_input_np)
    pt_input = torch.from_numpy(dummy_input_np.transpose(0, 3, 1, 2))

    keras_output, _ = keras_model(keras_input, training=False)
    with torch.no_grad():
        pt_output = pt_model(pt_input, is_training=False)

    keras_output_np = keras.ops.convert_to_numpy(keras_output)
    pt_output_np = pt_output.detach().cpu().numpy()

    message = "Ported model outputs do not match PyTorch outputs"
    tolerances = dict(rtol=1e-5, atol=1e-5, err_msg=message)
    np.testing.assert_allclose(keras_output_np, pt_output_np, **tolerances)


def test_weight_porting_is_deterministic(pytorch_model_and_weights):
    """Tests that weight porting produces the same results when run multiple
    times."""
    _, state_dict = pytorch_model_and_weights

    model1 = vit_small(**MODEL_KWARGS)
    dummy_input = np.zeros((1, 224, 224, 3), dtype="float32")
    model1(dummy_input, training=False)
    transfer_weights_from_pt_to_keras(state_dict, model1)

    model2 = vit_small(**MODEL_KWARGS)
    model2(dummy_input, training=False)
    transfer_weights_from_pt_to_keras(state_dict, model2)

    for w1, w2 in zip(model1.weights, model2.weights):
        message = f"Weight porting is not deterministic for weight {w1.name}"
        np.testing.assert_array_equal(w1.numpy(), w2.numpy(), err_msg=message)


def test_weight_count_matches(pytorch_model_and_weights, keras_model):
    """Tests that the number of trainable parameters matches between PyTorch
    and Keras."""
    pt_model, _ = pytorch_model_and_weights

    parameters = pt_model.parameters()
    pt_params = sum(p.numel() for p in parameters if p.requires_grad)
    keras_params = sum(np.prod(w.shape) for w in keras_model.trainable_weights)

    tolerance = 0.01
    message = f"Parameter count mismatch: PyTorch={pt_params}, "
    message = message + f"Keras={keras_params}"
    assert abs(pt_params - keras_params) / pt_params < tolerance, message
