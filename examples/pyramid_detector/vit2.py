from functools import partial
import keras
from keras import ops
from keras.layers import (
    Dense,
    Embedding,
    Layer,
    Input,
    Rescaling,
    LayerNormalization,
    Flatten,
    Dropout,
    Add,
    MultiHeadAttention,
)


class Patches(Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size, H, W, channels = ops.shape(images)
        num_patches_H = H // self.patch_size
        num_patches_W = W // self.patch_size
        patches = keras.ops.image.extract_patches(images, self.patch_size)
        num_channels = self.patch_size * self.patch_size * channels
        shape = (batch_size, num_patches_H * num_patches_W, num_channels)
        return ops.reshape(patches, shape)

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.project = Dense(units=projection_dim)
        self.position_embedding = Embedding(num_patches, projection_dim)

    def call(self, patch):
        positions = ops.expand_dims(ops.arange(0, self.num_patches), axis=0)
        return self.project(patch) + self.position_embedding(positions)

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config


def block_MLP(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=keras.activations.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x


def encoder(num_heads, dimension, units, encoded_patches):
    x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
    # key_dim = dimension // num_heads
    attention = MultiHeadAttention(num_heads, dimension, dropout=0.1)(x1, x1)
    x2 = Add()([attention, encoded_patches])
    x3 = LayerNormalization(epsilon=1e-6)(x2)
    x3 = block_MLP(x3, hidden_units=units, dropout_rate=0.1)
    encoded_patches = Add()([x3, x2])
    return encoded_patches


def ViT(
    input_shape,
    num_classes,
    patch_size,
    projection_dim=64,
    num_heads=4,
    transformer_units=[128, 64],
    transformer_layers=8,
    mlp_head_units=[2048, 1024],
):
    inputs = Input(shape=input_shape)
    inputs_normalized = Rescaling(1.0 / 255.0)(inputs)
    patches = Patches(patch_size)(inputs_normalized)
    num_patches = (input_shape[0] // patch_size) ** 2
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    encode = partial(encoder, num_heads, projection_dim, transformer_units)
    for _ in range(transformer_layers):
        encoded_patches = encode(encoded_patches)
    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = Flatten()(representation)
    representation = Dropout(0.5)(representation)
    features = block_MLP(representation, mlp_head_units, dropout_rate=0.5)
    logits = Dense(num_classes)(features)
    return keras.Model(inputs=inputs, outputs=logits)


if __name__ == "__main__":
    model = ViT((128, 128, 3), 1, 8)
