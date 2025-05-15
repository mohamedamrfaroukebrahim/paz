import os
import keras
from keras import layers
import jax.numpy as jp


def ViT(
    image_size=128,
    patch_size=16,
    num_transformer_layers=6,
    projection_dim=256,
    num_heads=4,
    transformer_mlp_dim=1024,  # Typically 4 * projection_dim
    head_units=[128],  # Hidden units in the classification head
    dropout_rate=0.1,
    num_classes=1,
):
    if projection_dim % num_heads != 0:
        raise ValueError(
            f"Projection dimension ({projection_dim}) must be divisible by "
            f"the number of heads ({num_heads})."
        )

    inputs = keras.Input(shape=(image_size, image_size, 3))

    num_patches = (image_size // patch_size) ** 2

    projected_patches = layers.Conv2D(
        filters=projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="patch_projection",
    )(inputs)

    projected_patches = layers.Reshape(
        target_shape=(num_patches, projection_dim),
        name="flatten_patches",
    )(projected_patches)

    class_token = keras.layers.Embedding(
        input_dim=1,
        output_dim=projection_dim,
        embeddings_initializer="glorot_uniform",
        name="cls_token_embedding",
    )(jp.zeros((keras.ops.shape(inputs)[0], 1), dtype="int32"))

    x = layers.Concatenate(axis=1, name="prepend_cls_token")(
        [class_token, projected_patches]
    )

    total_num_patches = num_patches + 1  # CLS token + all patchesh
    positional_embedding = layers.Embedding(
        input_dim=total_num_patches,
        output_dim=projection_dim,
        embeddings_initializer="glorot_uniform",
        name="positional_embedding",
    )

    positions = jp.arange(start=0, stop=total_num_patches)[jp.newaxis, :]
    positions = jp.broadcast_to(
        positions, (keras.ops.shape(x)[0], total_num_patches)
    )
    x = x + positional_embedding(positions)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="pos_emb_dropout")(x)

    # 3. Transformer Encoder Blocks
    for i in range(num_transformer_layers):
        # Layer Normalization 1
        x1 = layers.LayerNormalization(
            epsilon=1e-6, name=f"encoder_ln1_layer_{i}"
        )(x)
        # Multi-Head Attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim
            // num_heads,  # Dimension of each attention head's key/query/value
            dropout=dropout_rate,  # Dropout for attention weights
            name=f"multi_head_attention_layer_{i}",
        )(
            query=x1, value=x1, key=x1
        )  # Self-attention
        # Skip Connection 1
        x = layers.Add(name=f"encoder_skip1_layer_{i}")([x, attention_output])

        # Layer Normalization 2
        x2 = layers.LayerNormalization(
            epsilon=1e-6, name=f"encoder_ln2_layer_{i}"
        )(x)
        # MLP
        x2 = layers.Dense(
            units=transformer_mlp_dim,
            activation="gelu",
            name=f"encoder_mlp_dense1_layer_{i}",
        )(x2)
        if dropout_rate > 0:
            x2 = layers.Dropout(
                dropout_rate, name=f"encoder_mlp_dropout_layer_{i}"
            )(x2)
        x2 = layers.Dense(
            units=projection_dim, name=f"encoder_mlp_dense2_layer_{i}"
        )(
            x2
        )  # Project back to projection_dim
        # Skip Connection 2
        x = layers.Add(name=f"encoder_skip2_layer_{i}")([x, x2])

    # 4. Classification Head
    # Use the output of the CLS token
    class_token_output = x[:, 0, :]  # Shape: (batch_size, projection_dim)

    # Apply Layer Normalization before the MLP head
    processed_class_token = layers.LayerNormalization(
        epsilon=1e-6, name="head_ln"
    )(class_token_output)

    hidden_features = processed_class_token
    for arg, units in enumerate(head_units):
        hidden_features = layers.Dense(
            units, activation="gelu", name=f"head_dense_{arg}"
        )(hidden_features)
        if dropout_rate > 0:
            hidden_features = layers.Dropout(
                dropout_rate, name=f"head_dropout_{arg}"
            )(hidden_features)

    outputs = layers.Dense(num_classes, name="classifier_head")(hidden_features)
    return keras.Model(inputs=inputs, outputs=outputs, name="VIT")


if __name__ == "__main__":
    import os

    os.environ["KERAS_BACKEND"] = "jax"
    import jax.numpy as jp

    model = ViT(128, num_classes=1)
    logits = model(jp.zeros((10, 128, 128, 3)))
    print(logits.shape)
    model.summary()
