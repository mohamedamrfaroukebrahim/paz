import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from .dino_v2_blocks import build_vit_backbone, build_detection_head

VALID_BASE_WEIGHTS = [None]
VALID_HEAD_WEIGHTS = [None]


def DINOv2(image, num_classes, patch_size, embed_dim, depth, num_heads,
           mlp_ratio, drop_rate, attn_drop_rate, num_priors,
           num_register_tokens, model_name):
    """Creates a DINOv2-based object detection model.
    Uses a Vision Transformer (ViT) backbone inspired by DINOv2
    with a multi-scale detection head.

    # Arguments
        image: Tensor of shape `(batch_size, H, W, C)`, input image.
        num_classes: Int, number of object classes.
        patch_size: Int, patch size for patch embedding.
        embed_dim: Int, embedding dimensionality.
        depth: Int, number of transformer blocks.
        num_heads: Int, number of attention heads.
        mlp_ratio: Float, MLP hidden dim ratio.
        drop_rate: Float, dropout rate.
        attn_drop_rate: Float, attention dropout rate.
        num_priors: Int, number of anchor boxes per location.
        num_register_tokens: Int, number of register tokens.
        model_name: Str, name of the model.

    # Returns
        model: Keras Model.

    # References
        - [DINOv2: Learning Robust Visual Features without
          Supervision](https://arxiv.org/abs/2304.07193)
    """
    features, cls_output = build_vit_backbone(
        image, patch_size, embed_dim, depth, num_heads,
        mlp_ratio, drop_rate, attn_drop_rate, num_register_tokens)
    output = build_detection_head(features, num_classes, num_priors,
                                  embed_dim)
    model = Model(inputs=image, outputs=output, name=model_name)
    return model


def DINOv2ViTS(num_classes=80, input_shape=(560, 560, 3), num_priors=9,
               base_weights=None, head_weights=None,
               num_register_tokens=4, drop_rate=0.0, attn_drop_rate=0.0):
    """DINOv2 ViT-Small object detection model.

    # Arguments
        num_classes: Int, number of object classes.
        input_shape: Tuple of integers, input image shape (H, W, C).
        num_priors: Int, number of anchor boxes per patch.
        base_weights: Str or None, name of base weights to load.
            Currently only `None` is supported.
        head_weights: Str or None, name of head weights to load.
            Currently only `None` is supported.
        num_register_tokens: Int, number of register tokens.
        drop_rate: Float, dropout probability.
        attn_drop_rate: Float, attention dropout probability.

    # Returns
        model: Keras Model with ViT-S/14 DINOv2 backbone.
    """
    if base_weights not in VALID_BASE_WEIGHTS:
        raise ValueError('Invalid `base_weights`: {}'.format(base_weights))
    if head_weights not in VALID_HEAD_WEIGHTS:
        raise ValueError('Invalid `head_weights`: {}'.format(head_weights))
    image = Input(shape=input_shape, name='image')
    model = DINOv2(
        image=image,
        num_classes=num_classes,
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        num_priors=num_priors,
        num_register_tokens=num_register_tokens,
        model_name='dinov2-vits')
    return model


def DINOv2ViTB(num_classes=80, input_shape=(560, 560, 3), num_priors=9,
               base_weights=None, head_weights=None,
               num_register_tokens=4, drop_rate=0.0, attn_drop_rate=0.0):
    """DINOv2 ViT-Base object detection model.

    # Arguments
        num_classes: Int, number of object classes.
        input_shape: Tuple of integers, input image shape (H, W, C).
        num_priors: Int, number of anchor boxes per patch.
        base_weights: Str or None, name of base weights to load.
            Currently only `None` is supported.
        head_weights: Str or None, name of head weights to load.
            Currently only `None` is supported.
        num_register_tokens: Int, number of register tokens.
        drop_rate: Float, dropout probability.
        attn_drop_rate: Float, attention dropout probability.

    # Returns
        model: Keras Model with ViT-B/14 DINOv2 backbone.
    """
    if base_weights not in VALID_BASE_WEIGHTS:
        raise ValueError('Invalid `base_weights`: {}'.format(base_weights))
    if head_weights not in VALID_HEAD_WEIGHTS:
        raise ValueError('Invalid `head_weights`: {}'.format(head_weights))
    image = Input(shape=input_shape, name='image')
    model = DINOv2(
        image=image,
        num_classes=num_classes,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        num_priors=num_priors,
        num_register_tokens=num_register_tokens,
        model_name='dinov2-vitb')
    return model


def DINOv2ViTL(num_classes=80, input_shape=(560, 560, 3), num_priors=9,
               base_weights=None, head_weights=None,
               num_register_tokens=4, drop_rate=0.0, attn_drop_rate=0.0):
    """DINOv2 ViT-Large object detection model.

    # Arguments
        num_classes: Int, number of object classes.
        input_shape: Tuple of integers, input image shape (H, W, C).
        num_priors: Int, number of anchor boxes per patch.
        base_weights: Str or None, name of base weights to load.
            Currently only `None` is supported.
        head_weights: Str or None, name of head weights to load.
            Currently only `None` is supported.
        num_register_tokens: Int, number of register tokens.
        drop_rate: Float, dropout probability.
        attn_drop_rate: Float, attention dropout probability.

    # Returns
        model: Keras Model with ViT-L/14 DINOv2 backbone.
    """
    if base_weights not in VALID_BASE_WEIGHTS:
        raise ValueError('Invalid `base_weights`: {}'.format(base_weights))
    if head_weights not in VALID_HEAD_WEIGHTS:
        raise ValueError('Invalid `head_weights`: {}'.format(head_weights))
    image = Input(shape=input_shape, name='image')
    model = DINOv2(
        image=image,
        num_classes=num_classes,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        num_priors=num_priors,
        num_register_tokens=num_register_tokens,
        model_name='dinov2-vitl')
    return model
