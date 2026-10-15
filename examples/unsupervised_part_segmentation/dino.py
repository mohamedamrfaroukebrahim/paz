import keras.ops as ops


def compute_features(model, image):
    # Image is (H, W, 3). Add batch dimension.
    image_tensor = ops.expand_dims(image, axis=0)
    # The DINOv3 model returns (class_token, patch_tokens).
    _, features = model(image_tensor, training=False)
    # features shape: (B, N, D). Squeeze batch.
    features = ops.squeeze(features, axis=0)
    return ops.convert_to_numpy(features)
