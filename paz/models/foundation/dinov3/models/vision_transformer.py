import os

from keras import Input, Model, ops
from keras.layers import Concatenate

from paz.models.foundation.dinov3.layers.block import build_norm
from paz.models.foundation.dinov3.layers.block import (
    apply_self_attention_block,
)
from paz.models.foundation.dinov3.layers.patch_embed import apply_patch_embed
from paz.models.foundation.dinov3.layers.patch_embed import flatten_patches
from paz.models.foundation.dinov3.layers.patch_embed import make_2tuple
from paz.models.foundation.dinov3.layers.rope_position_encoding import (
    apply_rope_embed,
    compute_periods,
    validate_rope,
)
from paz.models.transformers.embeddings.token import LearnableTokens


def DinoVisionTransformer(*, img_size=224, patch_size=16, in_chans=3,
                          pos_embed_rope_base=100.0,
                          pos_embed_rope_min_period=None,
                          pos_embed_rope_max_period=None,
                          pos_embed_rope_normalize_coords="separate",
                          pos_embed_rope_shift_coords=None,
                          pos_embed_rope_jitter_coords=None,
                          pos_embed_rope_rescale_coords=None,
                          pos_embed_rope_dtype="bf16", embed_dim=768,
                          depth=12, num_heads=12, ffn_ratio=4.0,
                          qkv_bias=True, proj_bias=True, ffn_bias=True,
                          drop_path_rate=0.0, norm_layer="layernorm",
                          ffn_layer="mlp", layerscale_init=None,
                          n_storage_tokens=0, mask_k_bias=False,
                          untie_cls_and_patch_norms=False,
                          untie_global_and_local_cls_norm=False,
                          name="dinov3_vision_transformer", **ignored_kwargs):
    """DINOv3 ViT returning ``(class_token, patch_tokens)``.

    Takes channels-last images. The RoPE coordinate augmentations
    (shift, jitter, rescale) and ``untie_global_and_local_cls_norm`` act
    only on training crop lists, so the inference graph leaves them out.
    """
    if len(ignored_kwargs) > 0:
        print(f"Keras model ignored kwargs: {ignored_kwargs}")
    rope_periods = (pos_embed_rope_min_period, pos_embed_rope_max_period)
    rope_args = (pos_embed_rope_base, *rope_periods)
    validate_rope(embed_dim, num_heads, *rope_args)
    image_shape = make_2tuple(img_size) + (in_chans,)
    images = Input(image_shape, name="images")
    args = (images, make_2tuple(patch_size), embed_dim, True, "patch_embed")
    patches = apply_patch_embed(*args)
    height, width = patches.shape[1], patches.shape[2]
    patches = flatten_patches(patches, embed_dim, "patch_embed")
    tokens = prepend_tokens(patches, embed_dim, n_storage_tokens)
    head_dim = embed_dim // num_heads
    args = (height, width, head_dim, pos_embed_rope_normalize_coords)
    rope = apply_rope_embed(patches, *args, "rope_embed")
    num_prefix_tokens = n_storage_tokens + 1
    args = (embed_dim, num_heads, ffn_ratio, qkv_bias, proj_bias, ffn_bias)
    args = args + (0.0, 0.0, drop_path_rate, layerscale_init, "gelu")
    args = args + (norm_layer, ffn_layer, num_prefix_tokens)
    for index in range(depth):
        block_name = f"blocks_{index}"
        tokens = apply_self_attention_block(tokens, rope, *args, block_name)
    args = (norm_layer, num_prefix_tokens, untie_cls_and_patch_norms)
    outputs = normalize_tokens(tokens, *args)
    model = Model(images, outputs, name=name)
    assign_rope_periods(model, head_dim, *rope_args)
    return model


def prepend_tokens(patches, embed_dim, n_storage_tokens):
    class_token = LearnableTokens(1, embed_dim, name="cls_token")(patches)
    mask_token = LearnableTokens(1, embed_dim, name="mask_token")(patches)
    tokens = [class_token + 0 * mask_token]
    if n_storage_tokens > 0:
        args = (n_storage_tokens, embed_dim)
        storage = LearnableTokens(*args, name="storage_tokens")(patches)
        tokens.append(storage)
    tokens.append(patches)
    return Concatenate(axis=1, name="tokens")(tokens)


def normalize_tokens(tokens, norm_layer, num_prefix_tokens,
                     untie_cls_and_patch_norms):
    if untie_cls_and_patch_norms:
        prefix_tokens = tokens[:, :num_prefix_tokens]
        prefix_tokens = build_norm(norm_layer, "cls_norm")(prefix_tokens)
        patch_tokens = tokens[:, num_prefix_tokens:]
        patch_tokens = build_norm(norm_layer, "norm")(patch_tokens)
        return prefix_tokens[:, 0], patch_tokens
    normalized = build_norm(norm_layer, "norm")(tokens)
    return normalized[:, 0], normalized[:, num_prefix_tokens:]


def assign_rope_periods(model, head_dim, base, min_period, max_period):
    periods = compute_periods(head_dim, base, min_period, max_period)
    periods = ops.convert_to_numpy(periods)
    model.get_layer("rope_embed").set_weights([periods])


def load_pretrained_weights(model, model_name):
    filename = f"~/.keras/paz/models/{model_name}.keras"
    weights_path = os.path.expanduser(filename)
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        model.load_weights(weights_path)
    else:
        message = f"Weights file not found at {weights_path}."
        print(message + " Model initialized with random weights.")


def vit_small(patch_size=16, weights=None, input_shape=(224, 224, 3),
              **kwargs):
    defaults = {
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "ffn_ratio": 4,
        "n_storage_tokens": 4,
        "layerscale_init": 1e-6,
        "pos_embed_rope_dtype": "float32",
        "img_size": tuple(input_shape[:2]),
        "patch_size": patch_size,
    }
    defaults.update(kwargs)
    model = DinoVisionTransformer(**defaults)
    if weights == "pretrained":
        load_pretrained_weights(model, "dinov3_vits16_ported")
    return model


def vit_base(patch_size=16, weights=None, input_shape=(224, 224, 3),
             **kwargs):
    defaults = {
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "ffn_ratio": 4,
        "n_storage_tokens": 4,
        "layerscale_init": 1e-6,
        "pos_embed_rope_dtype": "float32",
        "img_size": tuple(input_shape[:2]),
        "patch_size": patch_size,
    }
    defaults.update(kwargs)
    model = DinoVisionTransformer(**defaults)
    if weights == "pretrained":
        load_pretrained_weights(model, "dinov3_vitb16_ported")
    return model


def vit_large(patch_size=16, weights=None, input_shape=(224, 224, 3),
              **kwargs):
    defaults = {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "ffn_ratio": 4,
        "n_storage_tokens": 4,
        "layerscale_init": 1e-6,
        "pos_embed_rope_dtype": "float32",
        "img_size": tuple(input_shape[:2]),
        "patch_size": patch_size,
    }
    defaults.update(kwargs)
    model = DinoVisionTransformer(**defaults)
    if weights == "pretrained":
        load_pretrained_weights(model, "dinov3_vitl16_ported")
    return model


def vit_so400m(patch_size=16, input_shape=(224, 224, 3), **kwargs):
    defaults = {
        "embed_dim": 1152,
        "depth": 27,
        "num_heads": 18,
        "ffn_ratio": 3.777777778,
        "img_size": tuple(input_shape[:2]),
        "patch_size": patch_size,
    }
    defaults.update(kwargs)
    model = DinoVisionTransformer(**defaults)
    return model


def vit_huge2(patch_size=16, input_shape=(224, 224, 3), **kwargs):
    defaults = {
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 20,
        "ffn_ratio": 4,
        "img_size": tuple(input_shape[:2]),
        "patch_size": patch_size,
    }
    defaults.update(kwargs)
    model = DinoVisionTransformer(**defaults)
    return model


def vit_giant2(patch_size=16, input_shape=(224, 224, 3), **kwargs):
    """Close to ViT-giant: embed-dim 1536 and 24 heads, so 64 per head."""
    defaults = {
        "embed_dim": 1536,
        "depth": 40,
        "num_heads": 24,
        "ffn_ratio": 4,
        "img_size": tuple(input_shape[:2]),
        "patch_size": patch_size,
    }
    defaults.update(kwargs)
    model = DinoVisionTransformer(**defaults)
    return model


def vit_7b(patch_size=16, input_shape=(224, 224, 3), **kwargs):
    defaults = {
        "embed_dim": 4096,
        "depth": 40,
        "num_heads": 32,
        "ffn_ratio": 3,
        "img_size": tuple(input_shape[:2]),
        "patch_size": patch_size,
    }
    defaults.update(kwargs)
    model = DinoVisionTransformer(**defaults)
    return model


def DINOV3VITS(input_shape=(224, 224, 3), **kwargs):
    return vit_small(weights="pretrained", input_shape=input_shape, **kwargs)


def DINOV3VITB(input_shape=(224, 224, 3), **kwargs):
    return vit_base(weights="pretrained", input_shape=input_shape, **kwargs)


def DINOV3VITL(input_shape=(224, 224, 3), **kwargs):
    return vit_large(weights="pretrained", input_shape=input_shape, **kwargs)
