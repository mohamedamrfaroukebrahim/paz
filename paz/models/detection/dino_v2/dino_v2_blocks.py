import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Concatenate
from .layers import (PatchEmbed, AddClassToken, AddRegisterTokens,
                     AddPositionEmbedding, TransformerBlock)


def build_vit_backbone(image, patch_size, embed_dim, depth, num_heads,
                       mlp_ratio, drop_rate, attn_drop_rate,
                       num_register_tokens=0):
    """Builds Vision Transformer backbone used in DINOv2.

    # Arguments
        image: Tensor of shape `(batch_size, H, W, C)`.
        patch_size: Int, patch size used for patch embedding.
        embed_dim: Int, embedding dimensionality.
        depth: Int, number of transformer blocks.
        num_heads: Int, number of attention heads.
        mlp_ratio: Float, ratio of MLP hidden dim to embed_dim.
        drop_rate: Float, dropout rate.
        attn_drop_rate: Float, attention dropout rate.
        num_register_tokens: Int, number of register tokens.

    # Returns
        features: List of Tensors, intermediate feature maps from
            selected transformer block outputs.
        cls_output: Tensor of shape `(batch_size, embed_dim)`.
    """
    image_h = image.shape[1]
    if image_h is not None:
        num_patches = (image_h // patch_size) ** 2
    else:
        num_patches = None

    x = PatchEmbed(patch_size, embed_dim, name='patch_embed')(image)
    x = AddClassToken(embed_dim, name='add_cls_token')(x)
    if num_register_tokens > 0:
        x = AddRegisterTokens(
            num_register_tokens, embed_dim,
            name='add_register_tokens')(x)
    num_extra = 1 + num_register_tokens
    if num_patches is not None:
        x = AddPositionEmbedding(
            num_patches, embed_dim,
            num_extra_tokens=num_extra,
            name='pos_embed')(x)

    features = []
    feature_indices = set([depth // 4 - 1, depth // 2 - 1,
                           depth * 3 // 4 - 1, depth - 1])
    for i in range(depth):
        x = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            name='blocks_{}'.format(i))(x)
        if i in feature_indices:
            features.append(x)

    x = LayerNormalization(epsilon=1e-6, name='norm')(x)
    cls_output = x[:, 0]
    return features, cls_output


def build_detection_head(features, num_classes, num_priors, embed_dim):
    """Builds a simple detection head on top of ViT features.
    Takes multi-scale features extracted from the ViT backbone and
    produces class logits and box regression outputs.

    # Arguments
        features: List of Tensors, intermediate ViT outputs.
        num_classes: Int, number of object classes.
        num_priors: Int, number of prior anchor boxes per location.
        embed_dim: Int, feature embedding dimension.

    # Returns
        output: Tensor of shape `(batch_size, num_boxes,
            num_priors * (num_classes + 4))`.
    """
    all_outputs = []
    for feature_idx, feature in enumerate(features):
        class_logits = Dense(
            num_priors * num_classes,
            name='cls_{}'.format(feature_idx))(feature)
        box_preds = Dense(
            num_priors * 4,
            name='box_{}'.format(feature_idx))(feature)
        combined = Concatenate(
            axis=-1,
            name='combined_{}'.format(feature_idx))([box_preds, class_logits])
        all_outputs.append(combined)

    output = Concatenate(axis=1, name='boxes')(all_outputs)
    return output

