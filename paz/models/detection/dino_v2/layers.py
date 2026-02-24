import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout


class PatchEmbed(Layer):
    """Patch embedding layer for Vision Transformer.
    Splits the input image into non-overlapping patches and projects
    each patch into an embedding vector using a convolution.

    # Arguments
        patch_size: Int, size of each square patch.
        embed_dim: Int, dimensionality of patch embedding.

    # Properties
        patch_size: Int.
        embed_dim: Int.
        projection: Conv2D layer.

    # Methods
        call()
    """

    def __init__(self, patch_size, embed_dim, **kwargs):
        super(PatchEmbed, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = tf.keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding='valid',
            name='projection')

    def call(self, x):
        x = self.projection(x)
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
        num_patches = height * width
        x = tf.reshape(x, (batch_size, num_patches, self.embed_dim))
        return x

    def get_config(self):
        config = super(PatchEmbed, self).get_config()
        config.update({'patch_size': self.patch_size,
                       'embed_dim': self.embed_dim})
        return config


class AddClassToken(Layer):
    """Prepends a trainable class token to a sequence of patch embeddings.

    # Arguments
        embed_dim: Int, dimensionality of the class token embedding.

    # Properties
        embed_dim: Int.
        cls_token: Trainable weight of shape `(1, 1, embed_dim)`.

    # Methods
        call()
    """

    def __init__(self, embed_dim, **kwargs):
        super(AddClassToken, self).__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name='cls_token',
            shape=(1, 1, self.embed_dim),
            initializer='zeros',
            trainable=True)
        super(AddClassToken, self).build(input_shape)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
        return tf.concat([cls_tokens, x], axis=1)

    def get_config(self):
        config = super(AddClassToken, self).get_config()
        config.update({'embed_dim': self.embed_dim})
        return config


class AddRegisterTokens(Layer):
    """Prepends trainable register tokens after the class token.

    # Arguments
        num_register_tokens: Int, number of register tokens.
        embed_dim: Int, dimensionality of the register token embedding.

    # Properties
        num_register_tokens: Int.
        embed_dim: Int.
        register_tokens: Trainable weight.

    # Methods
        call()
    """

    def __init__(self, num_register_tokens, embed_dim, **kwargs):
        super(AddRegisterTokens, self).__init__(**kwargs)
        self.num_register_tokens = num_register_tokens
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.register_tokens = self.add_weight(
            name='register_tokens',
            shape=(1, self.num_register_tokens, self.embed_dim),
            initializer='zeros',
            trainable=True)
        super(AddRegisterTokens, self).build(input_shape)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        reg_tokens = tf.tile(self.register_tokens, [batch_size, 1, 1])
        cls = x[:, :1, :]
        rest = x[:, 1:, :]
        return tf.concat([cls, reg_tokens, rest], axis=1)

    def get_config(self):
        config = super(AddRegisterTokens, self).get_config()
        config.update({'num_register_tokens': self.num_register_tokens,
                       'embed_dim': self.embed_dim})
        return config


class AddPositionEmbedding(Layer):
    """Adds fixed sine-cosine position embeddings to a token sequence.

    # Arguments
        num_patches: Int, number of image patches (excluding CLS token).
        embed_dim: Int, embedding dimensionality.
        num_extra_tokens: Int, number of extra tokens (CLS + register).

    # Properties
        pos_embed: Fixed position embedding tensor.

    # Methods
        call()
    """

    def __init__(self, num_patches, embed_dim, num_extra_tokens=1, **kwargs):
        super(AddPositionEmbedding, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.num_extra_tokens = num_extra_tokens
        pos_embed = build_sincos_position_embedding(num_patches, embed_dim)
        if num_extra_tokens > 1:
            cls_pos = pos_embed[:, :1, :]
            patch_pos = pos_embed[:, 1:, :]
            extra_pos = np.zeros(
                (1, num_extra_tokens - 1, embed_dim), dtype=np.float32)
            pos_embed = np.concatenate([cls_pos, extra_pos, patch_pos],
                                       axis=1)
        self.pos_embed = tf.constant(pos_embed, dtype=tf.float32)

    def call(self, x):
        return x + self.pos_embed

    def get_config(self):
        config = super(AddPositionEmbedding, self).get_config()
        config.update({'num_patches': self.num_patches,
                       'embed_dim': self.embed_dim,
                       'num_extra_tokens': self.num_extra_tokens})
        return config


class MultiHeadSelfAttention(Layer):
    """Multi-head self-attention layer for Vision Transformer.

    # Arguments
        embed_dim: Int, total embedding dimensionality.
        num_heads: Int, number of attention heads.
        attn_drop: Float, dropout rate applied to attention weights.
        proj_drop: Float, dropout rate applied after projection.

    # Properties
        embed_dim: Int.
        num_heads: Int.
        head_dim: Int.
        scale: Float.
        qkv: Dense layer.
        proj: Dense layer.
        attn_drop: Dropout layer.
        proj_drop: Dropout layer.

    # Methods
        call()
    """

    def __init__(self, embed_dim, num_heads, attn_drop=0.0,
                 proj_drop=0.0, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = Dense(3 * embed_dim, use_bias=True, name='qkv')
        self.proj = Dense(embed_dim, name='proj')
        self.attn_drop = Dropout(attn_drop)
        self.proj_drop = Dropout(proj_drop)

    def call(self, x, training=None):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (batch_size, seq_len, 3,
                                self.num_heads, self.head_dim))
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)
        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (batch_size, seq_len, self.embed_dim))
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x

    def get_config(self):
        config = super(MultiHeadSelfAttention, self).get_config()
        config.update({'embed_dim': self.embed_dim,
                       'num_heads': self.num_heads})
        return config


class MLP(Layer):
    """MLP block used in Vision Transformer.

    # Arguments
        embed_dim: Int, input dimensionality.
        mlp_ratio: Float, ratio of hidden to input dimensionality.
        drop: Float, dropout rate.

    # Properties
        fc1: Dense layer.
        fc2: Dense layer.
        drop: Dropout layer.

    # Methods
        call()
    """

    def __init__(self, embed_dim, mlp_ratio=4.0, drop=0.0, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = Dense(hidden_dim, activation='gelu', name='fc1')
        self.fc2 = Dense(embed_dim, name='fc2')
        self.drop = Dropout(drop)

    def call(self, x, training=None):
        x = self.fc1(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x

    def get_config(self):
        config = super(MLP, self).get_config()
        config.update({'embed_dim': self.embed_dim,
                       'mlp_ratio': self.mlp_ratio,
                       'drop': self.drop_rate})
        return config


class TransformerBlock(Layer):
    """Transformer block consisting of self-attention and MLP.

    # Arguments
        embed_dim: Int, embedding dimensionality.
        num_heads: Int, number of attention heads.
        mlp_ratio: Float, MLP hidden dimension ratio.
        drop: Float, dropout rate.
        attn_drop: Float, attention dropout rate.

    # Properties
        norm1: LayerNormalization.
        attn: MultiHeadSelfAttention.
        norm2: LayerNormalization.
        mlp: MLP.

    # Methods
        call()
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0,
                 drop=0.0, attn_drop=0.0, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop
        self.attn_drop_rate = attn_drop
        self.norm1 = LayerNormalization(epsilon=1e-6, name='norm1')
        self.attn = MultiHeadSelfAttention(
            embed_dim, num_heads, attn_drop, drop, name='attn')
        self.norm2 = LayerNormalization(epsilon=1e-6, name='norm2')
        self.mlp = MLP(embed_dim, mlp_ratio, drop, name='mlp')

    def call(self, x, training=None):
        x = x + self.attn(self.norm1(x), training=training)
        x = x + self.mlp(self.norm2(x), training=training)
        return x

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({'embed_dim': self.embed_dim,
                       'num_heads': self.num_heads,
                       'mlp_ratio': self.mlp_ratio,
                       'drop': self.drop_rate,
                       'attn_drop': self.attn_drop_rate})
        return config


def build_sincos_position_embedding(num_patches, embed_dim):
    """Builds fixed sine-cosine position embeddings.

    # Arguments
        num_patches: Int, number of image patches.
        embed_dim: Int, embedding dimensionality.

    # Returns
        pos_embed: Array of shape `(1, num_patches + 1, embed_dim)`.
    """
    position = np.arange(num_patches + 1, dtype=np.float32)
    d_model = embed_dim
    assert d_model % 2 == 0
    div_term = np.exp(
        np.arange(0, d_model, 2, dtype=np.float32) *
        -(np.log(10000.0) / d_model))
    pos_embed = np.zeros((1, num_patches + 1, d_model), dtype=np.float32)
    pos_embed[0, :, 0::2] = np.sin(position[:, np.newaxis] * div_term)
    pos_embed[0, :, 1::2] = np.cos(position[:, np.newaxis] * div_term)
    return pos_embed
