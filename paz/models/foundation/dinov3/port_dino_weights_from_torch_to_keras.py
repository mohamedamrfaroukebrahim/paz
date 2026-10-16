import os
import sys
import logging
import torch
import numpy as np

# --- Configuration for Keras Backend and Environment ---

os.environ["KERAS_BACKEND"] = "jax"
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch

# ==============================================================================
# Keras Layer Implementation
# ==============================================================================
from paz.models.foundation.dinov3.models.vision_transformer import (
    vit_small,
    vit_base,
    vit_large,
)
from paz.models.foundation.dinov3.models import convnext

# ==============================================================================
# PyTorch to Keras Weight Porting Function (Adapted for DINOv3)
# ==============================================================================


def transfer_weights_from_pt_to_keras(pt_state_dict, keras_model):
    """Loads a PyTorch DINOv3 ViT state_dict into a Keras DINOv3 ViT.

    Every target layer is resolved by name with ``get_layer``. Blocks are
    probed by name, so only the blocks the Keras model holds are ported.

    Args:
        pt_state_dict: The state_dict from the pretrained PyTorch model.
        keras_model: A ``keras.Model`` built by ``DinoVisionTransformer``.
    """
    logging.info(f"Starting weight porting for '{keras_model.name}'...")
    port_tokens(pt_state_dict, keras_model)
    port_patch_embed(pt_state_dict, keras_model)
    port_rope_periods(pt_state_dict, keras_model)
    block_index = 0
    while has_layer(keras_model, f"blocks_{block_index}_attn"):
        port_block(pt_state_dict, keras_model, block_index)
        block_index = block_index + 1
    logging.info(f"✓ Ported {block_index} transformer blocks.")
    port_final_norms(pt_state_dict, keras_model)
    logging.info("\n--- Weight porting complete! ---")
    return keras_model


def to_numpy(tensor):
    return tensor.detach().float().numpy()


def has_layer(model, name):
    try:
        model.get_layer(name)
    except ValueError:
        return False
    return True


def set_tokens(model, name, tensor):
    layer = model.get_layer(name)
    shape = tuple(layer.weights[0].shape)
    layer.set_weights([to_numpy(tensor).reshape(shape)])
    logging.info(f"✓ Ported {name}.")


def set_dense(layer, pt_state_dict, key):
    weights = [to_numpy(pt_state_dict[f"{key}.weight"]).T]
    if f"{key}.bias" in pt_state_dict:
        weights.append(to_numpy(pt_state_dict[f"{key}.bias"]))
    layer.set_weights(weights)


def set_norm(layer, pt_state_dict, key):
    weights = [to_numpy(pt_state_dict[f"{key}.weight"])]
    if f"{key}.bias" in pt_state_dict:
        weights.append(to_numpy(pt_state_dict[f"{key}.bias"]))
    layer.set_weights(weights)


def port_tokens(pt_state_dict, keras_model):
    set_tokens(keras_model, "cls_token", pt_state_dict["cls_token"])
    set_tokens(keras_model, "mask_token", pt_state_dict["mask_token"])
    has_storage = has_layer(keras_model, "storage_tokens")
    if "storage_tokens" in pt_state_dict and has_storage:
        storage_tokens = pt_state_dict["storage_tokens"]
        set_tokens(keras_model, "storage_tokens", storage_tokens)


def port_patch_embed(pt_state_dict, keras_model):
    kernel = pt_state_dict["patch_embed.proj.weight"].permute(2, 3, 1, 0)
    weights = [to_numpy(kernel)]
    if "patch_embed.proj.bias" in pt_state_dict:
        weights.append(to_numpy(pt_state_dict["patch_embed.proj.bias"]))
    keras_model.get_layer("patch_embed_proj").set_weights(weights)
    logging.info("✓ Ported patch_embed weights.")


def port_rope_periods(pt_state_dict, keras_model):
    if "rope_embed.periods" in pt_state_dict:
        periods = to_numpy(pt_state_dict["rope_embed.periods"])
        keras_model.get_layer("rope_embed").set_weights([periods])
        logging.info("✓ Ported rope_embed periods.")


def port_block(pt_state_dict, keras_model, index):
    prefix, name = f"blocks.{index}", f"blocks_{index}"
    norm1 = keras_model.get_layer(f"{name}_norm1")
    norm2 = keras_model.get_layer(f"{name}_norm2")
    set_norm(norm1, pt_state_dict, f"{prefix}.norm1")
    set_norm(norm2, pt_state_dict, f"{prefix}.norm2")
    attention = keras_model.get_layer(f"{name}_attn")
    qkv = attention.get_layer(f"{name}_attn_qkv")
    proj = attention.get_layer(f"{name}_attn_proj")
    set_dense(qkv, pt_state_dict, f"{prefix}.attn.qkv")
    set_dense(proj, pt_state_dict, f"{prefix}.attn.proj")
    port_layer_scales(pt_state_dict, keras_model, prefix, name)
    port_feedforward(pt_state_dict, keras_model, prefix, name)


def port_layer_scales(pt_state_dict, keras_model, prefix, name):
    for index in (1, 2):
        key = f"{prefix}.ls{index}.gamma"
        target = f"{name}_ls{index}"
        if key in pt_state_dict and has_layer(keras_model, target):
            gamma = to_numpy(pt_state_dict[key])
            keras_model.get_layer(target).set_weights([gamma])


def port_feedforward(pt_state_dict, keras_model, prefix, name):
    if f"{prefix}.mlp.fc1.weight" in pt_state_dict:
        parts = ("fc1", "fc2")
    elif f"{prefix}.mlp.w1.weight" in pt_state_dict:
        parts = ("w1", "w2", "w3")
    else:
        message = f"Could not find FFN weights for block {prefix}. "
        raise KeyError(message + "Neither MLP nor SwiGLUFFN keys exist.")
    feedforward = keras_model.get_layer(f"{name}_mlp")
    for part in parts:
        layer = feedforward.get_layer(f"{name}_mlp_{part}")
        set_dense(layer, pt_state_dict, f"{prefix}.mlp.{part}")
    logging.info(f"✓ Ported block {name}.")


def port_final_norms(pt_state_dict, keras_model):
    set_norm(keras_model.get_layer("norm"), pt_state_dict, "norm")
    logging.info("✓ Ported final 'norm' layer.")
    if has_layer(keras_model, "cls_norm"):
        cls_norm = keras_model.get_layer("cls_norm")
        set_norm(cls_norm, pt_state_dict, "cls_norm")
        logging.info("✓ Ported final 'cls_norm' layer.")


CONV_AXES = (2, 3, 1, 0)
DEPTHWISE_AXES = (2, 3, 0, 1)
DENSE_AXES = (1, 0)


def transfer_weights_convnext(pt_state_dict, keras_model):
    """Loads a PyTorch DINOv3 ConvNeXt state_dict into a Keras ConvNeXt.

    Keras layer names are the PyTorch keys with dots as underscores, and
    every target layer is resolved by name with ``get_layer``. Blocks are
    probed by name, so only the blocks the Keras model holds are ported.

    Args:
        pt_state_dict: The state_dict from the pretrained PyTorch model.
        keras_model: A ``keras.Model`` built by ``ConvNeXt``.
    """
    logging.info(f"Starting weight porting for '{keras_model.name}'...")
    port_downsampling(pt_state_dict, keras_model)
    num_blocks = 0
    for stage in range(4):
        index = 0
        while has_layer(keras_model, f"stages_{stage}_{index}_dwconv"):
            prefix = f"stages.{stage}.{index}"
            port_convnext_block(pt_state_dict, keras_model, prefix)
            index = index + 1
        num_blocks = num_blocks + index
    logging.info(f"✓ Ported {num_blocks} ConvNeXt blocks.")
    set_norm(keras_model.get_layer("norm"), pt_state_dict, "norm")
    logging.info("✓ Ported final 'norm' layer.")
    return keras_model


def port_downsampling(pt_state_dict, keras_model):
    port_kernel(pt_state_dict, keras_model, "downsample_layers.0.0", CONV_AXES)
    port_kernel(pt_state_dict, keras_model, "downsample_layers.0.1", None)
    for index in range(1, 4):
        prefix = f"downsample_layers.{index}"
        port_kernel(pt_state_dict, keras_model, f"{prefix}.0", None)
        port_kernel(pt_state_dict, keras_model, f"{prefix}.1", CONV_AXES)
    logging.info("✓ Ported downsample_layers weights.")


def port_convnext_block(pt_state_dict, keras_model, prefix):
    args = (pt_state_dict, keras_model)
    port_kernel(*args, f"{prefix}.dwconv", DEPTHWISE_AXES)
    port_kernel(*args, f"{prefix}.norm", None)
    port_kernel(*args, f"{prefix}.pwconv1", DENSE_AXES)
    port_kernel(*args, f"{prefix}.pwconv2", DENSE_AXES)
    gamma_name = to_layer_name(f"{prefix}.gamma")
    if has_layer(keras_model, gamma_name):
        gamma = to_numpy(pt_state_dict[f"{prefix}.gamma"])
        keras_model.get_layer(gamma_name).set_weights([gamma])


def port_kernel(pt_state_dict, keras_model, key, axes):
    kernel = pt_state_dict[f"{key}.weight"]
    kernel = kernel if axes is None else kernel.permute(*axes)
    bias = to_numpy(pt_state_dict[f"{key}.bias"])
    layer = keras_model.get_layer(to_layer_name(key))
    layer.set_weights([to_numpy(kernel), bias])


def to_layer_name(key):
    return key.replace(".", "_")


# ==============================================================================
# Execution Script
# ==============================================================================
# ==============================================================================
# Execution Script
# ==============================================================================

if __name__ == "__main__":

    DINO_REPO_PATH = ""  # Not needed for this porting
    MODELS_WEIGHTS_DIR_PATH = "/home/octavio/Storage/dinov3/"

    # --- ViT Model Definitions ---
    VIT_MODEL_CONSTRUCTORS = {
        "dinov3_vits16": vit_small,
        "dinov3_vitb16": vit_base,
        "dinov3_vitl16": vit_large,
    }
    VIT_MODEL_WEIGHTS_PATHS = {
        "dinov3_vits16": r"dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        "dinov3_vitb16": r"dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "dinov3_vitl16": r"dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
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

    # --- ConvNeXt Model Definitions ---
    CONVNEXT_MODEL_CONSTRUCTORS = {
        "dinov3_convnext_tiny": convnext.get_convnext_arch("convnext_tiny"),
        "dinov3_convnext_small": convnext.get_convnext_arch("convnext_small"),
        "dinov3_convnext_base": convnext.get_convnext_arch("convnext_base"),
        "dinov3_convnext_large": convnext.get_convnext_arch("convnext_large"),
    }
    CONVNEXT_MODEL_WEIGHTS_PATHS = {
        "dinov3_convnext_tiny": r"dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
        "dinov3_convnext_small": r"dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
        "dinov3_convnext_base": r"dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
        "dinov3_convnext_large": r"dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth",
    }
    convnext_model_kwargs = {
        "img_size": 224,  # Used for dummy input
        "patch_size": 16,  # Passed to constructor
        "layer_scale_init_value": 1e-6,  # Passed to constructor
    }

    # --- Combine All Model Configs ---
    MODEL_CONSTRUCTORS = {**VIT_MODEL_CONSTRUCTORS, **CONVNEXT_MODEL_CONSTRUCTORS}
    MODEL_WEIGHTS_PATHS = {**VIT_MODEL_WEIGHTS_PATHS, **CONVNEXT_MODEL_WEIGHTS_PATHS}

    output_dir = os.path.expanduser("~/.keras/paz/models/")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory set to: '{output_dir}'")

    MODEL_CONFIGS = {}
    for name, constructor in MODEL_CONSTRUCTORS.items():
        MODEL_CONFIGS[name] = {
            "constructor": constructor,
            "weight_filename": MODEL_WEIGHTS_PATHS[name],
            "keras_name": f"dino_{name}",
            # Select the correct kwargs dict based on model type
            "kwargs": vit_model_kwargs if "vit" in name else convnext_model_kwargs,
        }

    # --- Main Processing Loop ---
    for model_key, config in MODEL_CONFIGS.items():
        print("-" * 80)
        logging.info(f"Processing model: {model_key}")

        # --- Load PyTorch Weights ---
        weight_path = os.path.join(MODELS_WEIGHTS_DIR_PATH, config["weight_filename"])
        if not os.path.isfile(weight_path):
            logging.warning(
                f"PyTorch weight file not found at: {weight_path}. Skipping."
            )
            continue

        logging.info(f"Loading PyTorch state dict from: {weight_path}...")
        pytorch_state_dict = torch.load(weight_path, map_location=torch.device("cpu"))
        logging.info("✓ PyTorch weights loaded successfully.")

        # --- Instantiate Keras Model ---
        logging.info("Instantiating Keras 3 DINOv3 model architecture...")
        keras_dinov3_model = config["constructor"](
            name=config["keras_name"], **config["kwargs"]
        )

        # --- Build Keras Model ---
        logging.info("Building Keras model to initialize weights...")
        img_size = config["kwargs"]["img_size"]

        # *** IMPORTANT: Use correct input shape for model type ***
        if "convnext" in model_key:
            # ConvNeXt expects channels_first: (N, C, H, W)
            dummy_input = np.zeros((1, 3, img_size, img_size), dtype="float32")
        else:
            # ViT expects channels_last: (N, H, W, C)
            dummy_input = np.zeros((1, img_size, img_size, 3), dtype="float32")

        _ = keras_dinov3_model(dummy_input, training=False)
        logging.info("✓ Keras model built.")

        # --- Port Weights ---
        # *** IMPORTANT: Call the correct porting function ***
        if "convnext" in model_key:
            logging.info("Using ConvNeXt weight porting logic...")
            keras_dinov3_model_with_weights = transfer_weights_convnext(
                pytorch_state_dict, keras_dinov3_model
            )
        else:
            logging.info("Using ViT weight porting logic...")
            keras_dinov3_model_with_weights = transfer_weights_from_pt_to_keras(
                pytorch_state_dict, keras_dinov3_model
            )

        # --- Save Keras Model ---
        final_keras_model_path = os.path.join(output_dir, f"{model_key}_ported.keras")
        logging.info(
            f"Saving final native Keras model to '{final_keras_model_path}'..."
        )
        keras_dinov3_model_with_weights.save(final_keras_model_path)
        logging.info("✓ Save complete.")
        print("-" * 80 + "\n")

    print("All configured models have been successfully ported and saved.")
