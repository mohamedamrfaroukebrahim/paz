import pytest
import tensorflow as tf
from paz.models.detection.dino_v2 import DINOv2ViTS, DINOv2ViTB, DINOv2ViTL
from paz.models.detection.dino_v2.layers import (
    PatchEmbed, AddClassToken, AddRegisterTokens, AddPositionEmbedding,
    MultiHeadSelfAttention, MLP, TransformerBlock,
    build_sincos_position_embedding)
from paz.models.detection.dino_v2.dino_v2_blocks import (
    build_vit_backbone, build_detection_head)


def get_test_images(image_size, batch_size=1, channels=3):
    """Generates a simple mock image.

    # Arguments
        image_size: Int, H x W image resolution.
        batch_size: Int, number of images.
        channels: Int, number of image channels.

    # Returns
        Tensor of zeros with shape (batch_size, H, W, C).
    """
    return tf.zeros((batch_size, image_size, image_size, channels),
                    dtype=tf.float32)


@pytest.mark.parametrize(('image_size, patch_size, embed_dim, num_patches'),
                         [
                             (224, 14, 384, 256),
                             (280, 14, 384, 400),
                             (560, 14, 384, 1600),
                         ])
def test_patch_embed_output_shape(image_size, patch_size, embed_dim,
                                  num_patches):
    images = get_test_images(image_size)
    patch_embed = PatchEmbed(patch_size=patch_size, embed_dim=embed_dim)
    output = patch_embed(images)
    expected_shape = (1, num_patches, embed_dim)
    assert output.shape == expected_shape, (
        f'PatchEmbed output shape mismatch: '
        f'expected {expected_shape}, got {output.shape}')
    del images, output


@pytest.mark.parametrize(('seq_len, embed_dim, num_heads'),
                         [
                             (64, 384, 6),
                             (256, 384, 6),
                             (64, 768, 12),
                         ])
def test_multi_head_self_attention_output_shape(seq_len, embed_dim,
                                                num_heads):
    x = tf.zeros((1, seq_len, embed_dim))
    attn = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
    output = attn(x)
    assert output.shape == (1, seq_len, embed_dim), (
        'MultiHeadSelfAttention output shape mismatch')
    del x, output


@pytest.mark.parametrize(('seq_len, embed_dim, mlp_ratio'),
                         [
                             (64, 384, 4.0),
                             (256, 768, 4.0),
                         ])
def test_mlp_output_shape(seq_len, embed_dim, mlp_ratio):
    x = tf.zeros((1, seq_len, embed_dim))
    mlp = MLP(embed_dim=embed_dim, mlp_ratio=mlp_ratio)
    output = mlp(x)
    assert output.shape == (1, seq_len, embed_dim), (
        'MLP output shape mismatch')
    del x, output


@pytest.mark.parametrize(('seq_len, embed_dim, num_heads'),
                         [
                             (64, 384, 6),
                             (256, 384, 6),
                         ])
def test_transformer_block_output_shape(seq_len, embed_dim, num_heads):
    x = tf.zeros((1, seq_len, embed_dim))
    block = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads)
    output = block(x)
    assert output.shape == (1, seq_len, embed_dim), (
        'TransformerBlock output shape mismatch')
    del x, output


@pytest.mark.parametrize(('num_patches, embed_dim'),
                         [
                             (256, 384),
                             (400, 768),
                         ])
def test_sincos_position_embedding_shape(num_patches, embed_dim):
    pos_embed = build_sincos_position_embedding(num_patches, embed_dim)
    expected_shape = (1, num_patches + 1, embed_dim)
    assert pos_embed.shape == expected_shape, (
        f'Position embedding shape mismatch: '
        f'expected {expected_shape}, got {pos_embed.shape}')
    del pos_embed


@pytest.mark.parametrize(('model_fn, model_name, embed_dim'),
                         [
                             (DINOv2ViTS, 'dinov2-vits', 384),
                             (DINOv2ViTB, 'dinov2-vitb', 768),
                         ])
def test_model_instantiation(model_fn, model_name, embed_dim):
    model = model_fn(num_classes=80, input_shape=(560, 560, 3))
    assert model.name == model_name, (
        f'Model name mismatch: expected {model_name}, got {model.name}')
    assert model.input.name == 'image', (
        'Model input name should be "image"')
    del model


@pytest.mark.parametrize(('model_fn, image_size, num_classes, num_priors'),
                         [
                             (DINOv2ViTS, 560, 80, 9),
                             (DINOv2ViTS, 280, 80, 9),
                         ])
def test_model_output_shape(model_fn, image_size, num_classes, num_priors):
    model = model_fn(
        num_classes=num_classes,
        input_shape=(image_size, image_size, 3),
        num_priors=num_priors)
    images = get_test_images(image_size)
    output = model(images)
    assert len(output.shape) == 3, 'Output should be 3D tensor'
    assert output.shape[0] == 1, 'Batch size mismatch'
    expected_last_dim = num_priors * (num_classes + 4)
    assert output.shape[-1] == expected_last_dim, (
        f'Output last dim mismatch: expected {expected_last_dim}, '
        f'got {output.shape[-1]}')
    del model, images, output
