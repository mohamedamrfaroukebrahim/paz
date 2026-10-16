import os
import sys
import torch
import numpy as np
import pytest
import keras
from keras import ops

from paz.models.foundation.dinov3.models.vision_transformer import (
    vit_small,
    vit_base,
    vit_large,
)
from paz.models.foundation.dinov3.models import convnext

from paz.models.foundation.dinov3.models.torch_vision_transformer_for_testing import (
    PT_vit_small,
    PT_vit_base,
    PT_vit_large,
)
from paz.models.foundation.dinov3.models.torch_convnext_for_testing import (
    PT_get_convnext_arch,
)


DINO_REPO_PATH = ""
PT_MODELS_WEIGHTS_DIR_PATH = "/home/octavio/Storage/dinov3/"
KERAS_MODELS_WEIGHTS_DIR_PATH = os.path.expanduser("~/.keras/paz/models/")

VIT_MODEL_WEIGHTS_PATHS = {
    "dinov3_vits16": r"dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "dinov3_vitb16": r"dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitl16": r"dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
}
KERAS_VIT_MODEL_WEIGHTS_PATHS = {
    "dinov3_vits16": r"dinov3_vits16_ported.keras",
    "dinov3_vitb16": r"dinov3_vitb16_ported.keras",
    "dinov3_vitl16": r"dinov3_vitl16_ported.keras",
}
vit_model_kwargs = {
    "img_size": 224,
    "patch_size": 16,
    "ffn_layer": "mlp",
    "untie_cls_and_patch_norms": False,
    "norm_layer": "layernorm",
    "layerscale_init": 1e-6,
    "n_storage_tokens": 4,
    "pos_embed_rope_dtype": "float32",
}

CONVNEXT_MODEL_WEIGHTS_PATHS = {
    "dinov3_convnext_tiny":
        r"dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
    "dinov3_convnext_small":
        r"dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
    "dinov3_convnext_base":
        r"dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
    "dinov3_convnext_large":
        r"dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth",
}
KERAS_CONVNEXT_MODEL_WEIGHTS_PATHS = {
    "dinov3_convnext_tiny": r"dinov3_convnext_tiny_ported.keras",
    "dinov3_convnext_small": r"dinov3_convnext_small_ported.keras",
    "dinov3_convnext_base": r"dinov3_convnext_base_ported.keras",
    "dinov3_convnext_large": r"dinov3_convnext_large_ported.keras",
}
convnext_model_kwargs = {
    "img_size": 224,
    "patch_size": 16,
    "layer_scale_init_value": 1e-6,
}
PT_CONVNEXT_CONSTRUCTORS = {
    "dinov3_convnext_tiny": PT_get_convnext_arch("convnext_tiny"),
    "dinov3_convnext_small": PT_get_convnext_arch("convnext_small"),
    "dinov3_convnext_base": PT_get_convnext_arch("convnext_base"),
    "dinov3_convnext_large": PT_get_convnext_arch("convnext_large"),
}
KERAS_CONVNEXT_CONSTRUCTORS = {
    "dinov3_convnext_tiny": convnext.get_convnext_arch("convnext_tiny"),
    "dinov3_convnext_small": convnext.get_convnext_arch("convnext_small"),
    "dinov3_convnext_base": convnext.get_convnext_arch("convnext_base"),
    "dinov3_convnext_large": convnext.get_convnext_arch("convnext_large"),
}

PT_MODEL_WEIGHTS_PATHS = {
    **VIT_MODEL_WEIGHTS_PATHS,
    **CONVNEXT_MODEL_WEIGHTS_PATHS,
}
KERAS_MODEL_WEIGHTS_PATHS = {
    **KERAS_VIT_MODEL_WEIGHTS_PATHS,
    **KERAS_CONVNEXT_MODEL_WEIGHTS_PATHS,
}

VIT_MODEL_CONFIGS = [
    (PT_vit_small, vit_small, "dinov3_vits16", vit_model_kwargs),
    (PT_vit_base, vit_base, "dinov3_vitb16", vit_model_kwargs),
    (PT_vit_large, vit_large, "dinov3_vitl16", vit_model_kwargs),
]

CONVNEXT_MODEL_CONFIGS = [
    (
        PT_CONVNEXT_CONSTRUCTORS[key],
        KERAS_CONVNEXT_CONSTRUCTORS[key],
        key,
        convnext_model_kwargs,
    )
    for key in KERAS_CONVNEXT_CONSTRUCTORS
]

ALL_MODEL_CONFIGS = VIT_MODEL_CONFIGS + CONVNEXT_MODEL_CONFIGS

CONFIG_ARGNAMES = "pt_constructor, keras_constructor, weight_key, model_kwargs"


def get_PT_model(pt_constructor, pt_weight_path_key, model_kwargs):
    """Instantiates a PyTorch model and loads its pre-trained weights."""
    if DINO_REPO_PATH not in sys.path:
        sys.path.insert(0, DINO_REPO_PATH)

    filename = PT_MODEL_WEIGHTS_PATHS[pt_weight_path_key]
    pt_weight_path = os.path.join(PT_MODELS_WEIGHTS_DIR_PATH, filename)
    if not os.path.isfile(pt_weight_path):
        pytest.skip(f"PyTorch weight file not found: {pt_weight_path}")

    pt_model = pt_constructor(**model_kwargs)
    state_dict = torch.load(pt_weight_path, map_location=torch.device("cpu"))
    pt_model.load_state_dict(state_dict, strict=False)
    pt_model.eval()
    print(f"PyTorch model '{pt_weight_path_key}' loaded from {pt_weight_path}")
    return pt_model


def get_keras_model(keras_constructor, keras_weight_path_key, model_kwargs):
    """Instantiates a Keras model, builds it, and loads its ported weights."""
    filename = KERAS_MODEL_WEIGHTS_PATHS[keras_weight_path_key]
    keras_weight_path = os.path.join(KERAS_MODELS_WEIGHTS_DIR_PATH, filename)
    if not os.path.isfile(keras_weight_path):
        pytest.skip(f"Keras weight file not found: {keras_weight_path}")

    keras_model = keras_constructor(**model_kwargs)

    # --- Build the model by passing a dummy input ---
    img_size = model_kwargs["img_size"]
    if "convnext" in keras_weight_path_key:
        # ConvNeXt expects channels_first: (N, C, H, W)
        shape = (1, 3, img_size, img_size)
    else:
        # ViT expects channels_last: (N, H, W, C)
        shape = (1, img_size, img_size, 3)
    dummy_input = np.random.randn(*shape).astype("float32")

    keras_model(dummy_input, training=False)

    # Load the pre-ported weights
    keras_model.load_weights(keras_weight_path)
    message = f"Keras model '{keras_weight_path_key}' loaded from "
    print(message + keras_weight_path)
    return keras_model


def count_blocks(keras_model):
    names = [layer.name for layer in keras_model.layers]
    return sum(name.endswith("_attn") for name in names)


def get_block(keras_model, index):
    def get_block_layer(part):
        return keras_model.get_layer(f"blocks_{index}_{part}")
    return get_block_layer


def build_probe(keras_model):
    """Exposes the prepared tokens and the RoPE (sin, cos) tables."""
    names = ("tokens", "rope_embed_sin", "rope_embed_cos")
    outputs = [keras_model.get_layer(name).output for name in names]
    return keras.Model(keras_model.inputs, outputs)


def has_layer(model, name):
    return name in [layer.name for layer in model.layers]


def count_stage_blocks(keras_model, stage):
    names = [layer.name for layer in keras_model.layers]
    prefix = f"stages_{stage}_"
    is_block = [name.startswith(prefix) for name in names if "dwconv" in name]
    return sum(is_block)


def get_convnext_block(keras_model, stage, index):
    def get_block_layer(part):
        return keras_model.get_layer(f"stages_{stage}_{index}_{part}")
    return get_block_layer


def apply_downsample(keras_model, index, x):
    for part in (0, 1):
        layer = keras_model.get_layer(f"downsample_layers_{index}_{part}")
        x = layer(x, training=False)
    return x


def apply_drop_path(keras_model, stage, index, x):
    name = f"stages_{stage}_{index}_drop_path"
    if has_layer(keras_model, name):
        x = keras_model.get_layer(name)(x, training=False)
    return x


def print_differences(keras_output_np, pt_output_np):
    difference = np.abs(keras_output_np - pt_output_np)
    print(f"Max absolute difference: {np.max(difference)}")
    print(f"Mean absolute difference: {np.mean(difference)}")


@pytest.mark.parametrize(CONFIG_ARGNAMES, VIT_MODEL_CONFIGS)
def test_final_VIT_output_equivalence(
    pt_constructor, keras_constructor, weight_key, model_kwargs
):
    print(f"\n--- Testing Final Output Equivalence: {weight_key} ---")
    pt_model = get_PT_model(pt_constructor, weight_key, model_kwargs)
    keras_model = get_keras_model(keras_constructor, weight_key, model_kwargs)

    # --- Create Identical Inputs ---
    np.random.seed(42)  # Use a fixed seed for reproducibility
    dummy_input_np = np.random.randn(1, 224, 224, 3).astype("float32")
    keras_input = keras.ops.convert_to_tensor(dummy_input_np)
    pt_input = torch.from_numpy(dummy_input_np.transpose(0, 3, 1, 2))
    print("Generated identical inputs.")

    # --- Get Model Outputs ---
    print("Performing inference...")
    keras_output, _ = keras_model(keras_input, training=False)
    with torch.no_grad():
        pt_output = pt_model(pt_input, is_training=False)

    # --- Compare Outputs ---
    print("Comparing outputs...")
    keras_output_np = keras.ops.convert_to_numpy(keras_output)
    pt_output_np = pt_output.detach().cpu().numpy()

    try:
        message = "Final model outputs do not match after loading weights"
        tolerances = dict(rtol=1e-5, atol=1e-5, err_msg=message)
        np.testing.assert_allclose(keras_output_np, pt_output_np, **tolerances)
    except AssertionError as e:
        print("\n❌ Outputs do not match!")
        print_differences(keras_output_np, pt_output_np)
        raise e
    print(f"✅ Success! Final outputs for {weight_key} match.")


@pytest.mark.parametrize(CONFIG_ARGNAMES, CONVNEXT_MODEL_CONFIGS)
def test_final_CONVNEXT_output_equivalence(
    pt_constructor, keras_constructor, weight_key, model_kwargs
):
    print(f"\n--- Testing Final Output Equivalence: {weight_key} ---")
    pt_model = get_PT_model(pt_constructor, weight_key, model_kwargs)
    keras_model = get_keras_model(keras_constructor, weight_key, model_kwargs)

    # --- Create Identical Inputs ---
    np.random.seed(42)  # Use a fixed seed for reproducibility

    if "convnext" in weight_key:
        print("Using channels-first input for ConvNeXt.")
        dummy_input_np = np.random.randn(1, 3, 224, 224).astype("float32")
        keras_input = keras.ops.convert_to_tensor(dummy_input_np)
        pt_input = torch.from_numpy(dummy_input_np)
    else:
        print("Using channels-last input for ViT.")
        dummy_input_np = np.random.randn(1, 224, 224, 3).astype("float32")
        keras_input = keras.ops.convert_to_tensor(dummy_input_np)
        pt_input = torch.from_numpy(dummy_input_np.transpose(0, 3, 1, 2))

    print("Generated identical inputs.")

    # --- Get Model Outputs ---
    print("Performing inference...")
    keras_output, _ = keras_model(keras_input, training=False)
    with torch.no_grad():
        pt_output = pt_model(pt_input, is_training=False)

    # --- Compare Outputs ---
    print("Comparing outputs...")
    keras_output_np = keras.ops.convert_to_numpy(keras_output)
    pt_output_np = pt_output.detach().cpu().numpy()

    try:
        mean_diff = np.mean(np.abs(keras_output_np - pt_output_np))
        print(f"Mean absolute difference: {mean_diff}")
        message = f"Mean absolute difference {mean_diff} exceeds "
        assert mean_diff < 1e-5, message + "tolerance of 1e-5."

    except AssertionError as e:
        print("\n❌ Outputs do not match!")
        print_differences(keras_output_np, pt_output_np)
        raise e
    print(f"✅ Success! Final outputs for {weight_key} match.")


@pytest.mark.parametrize(CONFIG_ARGNAMES, VIT_MODEL_CONFIGS)
def test_deep_dive_VIT_block_equivalence(
    pt_constructor, keras_constructor, weight_key, model_kwargs
):
    print(f"\n--- Testing Deep-Dive Block Equivalence: {weight_key} ---")
    pt_model = get_PT_model(pt_constructor, weight_key, model_kwargs)
    keras_model = get_keras_model(keras_constructor, weight_key, model_kwargs)

    any_failure = False

    def _compare_and_correct(keras_tensor, torch_tensor, layer_name, block_idx):
        nonlocal any_failure
        torch_np = torch_tensor.detach().cpu().numpy()
        keras_np = keras.ops.convert_to_numpy(keras_tensor)
        label = f"Block {block_idx} - {layer_name}"
        try:
            mean_diff = np.mean(np.abs(keras_np - torch_np))
            print(f"    Comparing {label}: Mean abs diff = {mean_diff}")
            if np.isnan(keras_np).any():
                raise AssertionError("Keras output contains NaN values.")
            if mean_diff > 2e-4:
                message = f"Mean absolute difference {mean_diff} exceeds "
                raise AssertionError(message + "tolerance of 2e-4.")
            print(f"    ✅ {label}: Output matches.")
            return keras_tensor
        except AssertionError as e:
            print(f"    ❌ FAILURE at {label}: Output mismatch.")
            print(f"       {e}")
            any_failure = True
            print("       Correcting Keras tensor to continue analysis...")
            return keras.ops.convert_to_tensor(torch_np)

    check = _compare_and_correct
    probe = build_probe(keras_model)
    for i in range(count_blocks(keras_model)):
        print(f"\n--- Analyzing Block {i} ---")

        # 1. Create a new, unique input for this block to isolate errors
        np.random.seed(i)
        img_size = model_kwargs["img_size"]
        image_np = np.random.randn(1, img_size, img_size, 3).astype("float32")
        keras_input = keras.ops.convert_to_tensor(image_np)
        pt_input = torch.from_numpy(image_np.transpose(0, 3, 1, 2))

        # 2. Get initial tokens
        # 3. ... together with the RoPE embeddings
        keras_tokens, *rope_keras = probe(keras_input)
        pt_tokens, (H_pt, W_pt) = pt_model.prepare_tokens_with_masks(pt_input)
        rope_torch = pt_model.rope_embed(H=H_pt, W=W_pt)

        # 4. Get the corresponding block from each model
        keras_block = get_block(keras_model, i)
        pt_block = pt_model.blocks[i]

        # --- Deep Dive into the Block ---
        keras_norm1_out = keras_block("norm1")(keras_tokens)
        pt_norm1_out = pt_block.norm1(pt_tokens)
        keras_norm1_out = check(keras_norm1_out, pt_norm1_out, "Norm1", i)

        attn_inputs = [keras_norm1_out, *rope_keras]
        keras_attn_out = keras_block("attn")(attn_inputs, training=False)
        pt_attn_out = pt_block.attn(pt_norm1_out, rope=rope_torch)
        keras_attn_out = check(keras_attn_out, pt_attn_out, "Attention", i)

        keras_ls1_out = keras_block("ls1")(keras_attn_out)
        pt_ls1_out = pt_block.ls1(pt_attn_out)
        keras_ls1_out = check(keras_ls1_out, pt_ls1_out, "LayerScale1", i)

        keras_res1_out = keras_tokens + keras_ls1_out
        pt_res1_out = pt_tokens + pt_ls1_out
        keras_res1_out = check(keras_res1_out, pt_res1_out, "Residual1", i)

        keras_norm2_out = keras_block("norm2")(keras_res1_out)
        pt_norm2_out = pt_block.norm2(pt_res1_out)
        keras_norm2_out = check(keras_norm2_out, pt_norm2_out, "Norm2", i)

        keras_mlp_out = keras_block("mlp")(keras_norm2_out, training=False)
        pt_mlp_out = pt_block.mlp(pt_norm2_out)
        keras_mlp_out = check(keras_mlp_out, pt_mlp_out, "MLP", i)

        keras_ls2_out = keras_block("ls2")(keras_mlp_out)
        pt_ls2_out = pt_block.ls2(pt_mlp_out)
        keras_ls2_out = check(keras_ls2_out, pt_ls2_out, "LayerScale2", i)

        keras_final_out = keras_res1_out + keras_ls2_out
        pt_final_out = pt_res1_out + pt_ls2_out
        check(keras_final_out, pt_final_out, "Final Block Output", i)

    print(f"\n--- Deep-Dive Analysis for {weight_key} Complete ---")
    message = f"One or more internal layers for {weight_key} had an output "
    assert not any_failure, message + "mismatch. See logs for details."


@pytest.mark.parametrize(CONFIG_ARGNAMES, CONVNEXT_MODEL_CONFIGS)
def test_deep_dive_convnext_block_equivalence(
    pt_constructor, keras_constructor, weight_key, model_kwargs
):
    print(f"\n--- Testing Deep-Dive Block Equivalence: {weight_key} ---")
    pt_model = get_PT_model(pt_constructor, weight_key, model_kwargs)
    keras_model = get_keras_model(keras_constructor, weight_key, model_kwargs)

    any_failure = False

    # Use the same comparison helper as the ViT test
    def _compare_and_correct(keras_tensor, torch_tensor, layer_name, block_str):
        nonlocal any_failure
        torch_np = torch_tensor.detach().cpu().numpy()
        keras_np = keras.ops.convert_to_numpy(keras_tensor)
        label = f"{block_str} - {layer_name}"
        try:
            mean_diff = np.mean(np.abs(keras_np - torch_np))
            print(f"    Comparing {label}: Mean abs diff = {mean_diff}")
            if np.isnan(keras_np).any():
                raise AssertionError("Keras output contains NaN values.")
            if mean_diff > 1e-4:
                message = f"Mean absolute difference {mean_diff} exceeds "
                raise AssertionError(message + "tolerance of 1e-4.")
            print(f"    ✅ {label}: Output matches.")
            return keras.ops.convert_to_tensor(torch_np)
        except AssertionError as e:
            print(f"    ❌ FAILURE at {label}: Output mismatch.")
            print(f"       {e}")
            any_failure = True
            print("       Correcting Keras tensor to continue analysis...")
            return keras.ops.convert_to_tensor(torch_np)

    check = _compare_and_correct
    np.random.seed(42)
    np_input = np.random.rand(2, 3, 224, 224).astype("float32")
    x_keras = keras.ops.convert_to_tensor(np_input)
    x_torch = torch.from_numpy(np_input)

    for i in range(4):
        print(f"\n--- Analyzing Stage {i} ({weight_key}) ---")

        # --- 1. Downsample Layer ---
        pt_down_layer = pt_model.downsample_layers[i]

        x_keras = apply_downsample(keras_model, i, x_keras)
        x_torch = pt_down_layer(x_torch)

        x_keras = check(x_keras, x_torch, "Downsample Output", f"Stage {i}")

        # --- 2. Blocks in Stage ---
        for j in range(count_stage_blocks(keras_model, i)):
            keras_block = get_convnext_block(keras_model, i, j)
            pt_block = pt_model.stages[i][j]

            input_keras = x_keras
            input_torch = x_torch

            block_prefix = f"Stage {i} Block {j}"

            # 1. DwConv
            k_dwconv = keras_block("dwconv")(input_keras)
            pt_dwconv = pt_block.dwconv(input_torch)
            k_dwconv = check(k_dwconv, pt_dwconv, "dwconv", block_prefix)

            # 2. Permute
            k_perm1 = ops.transpose(k_dwconv, (0, 2, 3, 1))
            pt_perm1 = pt_dwconv.permute(0, 2, 3, 1)

            # 3. Norm
            k_norm = keras_block("norm")(k_perm1)
            pt_norm = pt_block.norm(pt_perm1)
            k_norm = check(k_norm, pt_norm, "norm", block_prefix)

            # 4. PwConv1
            k_pw1 = keras_block("pwconv1")(k_norm)
            pt_pw1 = pt_block.pwconv1(pt_norm)
            k_pw1 = check(k_pw1, pt_pw1, "pwconv1", block_prefix)

            # 5. Act
            k_act = keras_block("act")(k_pw1)
            pt_act = pt_block.act(pt_pw1)
            k_act = check(k_act, pt_act, "act", block_prefix)

            # 6. PwConv2
            k_pw2 = keras_block("pwconv2")(k_act)
            pt_pw2 = pt_block.pwconv2(pt_act)
            k_pw2 = check(k_pw2, pt_pw2, "pwconv2", block_prefix)

            # 7. Gamma (LayerScale)
            if has_layer(keras_model, f"stages_{i}_{j}_gamma"):
                k_pw2_scaled = keras_block("gamma")(k_pw2)
                pt_pw2_scaled = pt_block.gamma * pt_pw2
                args = (k_pw2_scaled, pt_pw2_scaled, "gamma", block_prefix)
                k_pw2_scaled = check(*args)
            else:
                k_pw2_scaled = k_pw2
                pt_pw2_scaled = pt_pw2

            # 8. Permute Back
            k_perm2 = ops.transpose(k_pw2_scaled, (0, 3, 1, 2))
            pt_perm2 = pt_pw2_scaled.permute(0, 3, 1, 2)

            # 9. DropPath (is Identity in eval mode)
            k_drop = apply_drop_path(keras_model, i, j, k_perm2)
            pt_drop = pt_block.drop_path(pt_perm2)
            k_drop = check(k_drop, pt_drop, "drop_path", block_prefix)

            # 10. Residual
            x_keras = input_keras + k_drop
            x_torch = input_torch + pt_drop
            x_keras = check(x_keras, x_torch, "residual_add", block_prefix)

    print(f"\n--- Deep-Dive Analysis Complete for: {weight_key} ---")
    message = f"One or more internal layers for {weight_key} had an output "
    assert not any_failure, message + "mismatch. See logs for details."
