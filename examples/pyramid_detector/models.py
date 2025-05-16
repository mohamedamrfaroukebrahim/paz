import keras
from keras import layers


def SimpleCNN(shape, num_classes):
    return keras.Sequential(
        [
            keras.Input(shape=shape),
            keras.layers.Rescaling(scale=1 / 127.5, offset=-1),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(num_classes),
        ]
    )


def FineTuneXception(shape, num_classes):
    base = keras.applications.Xception(
        weights="imagenet",
        input_shape=shape,
        include_top=False,
    )
    base.trainable = False

    inputs = keras.Input(shape=(128, 128, 3))
    scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    x = scale_layer(inputs)

    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes)(x)
    return keras.Model(inputs, outputs)


def ConvNeXtTiny(input_shape, num_classes, dropout_rate=0.2):
    base = keras.applications.ConvNeXtTiny(
        include_top=False,
        include_preprocessing=True,
        # weights="imagenet",
        weights=None,
        input_shape=input_shape,
    )
    base.trainable = False
    inputs = keras.Input(shape=(128, 128, 3))
    x = base(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(num_classes)(x)
    return keras.Model(inputs, outputs)


def MiniXception(
    input_shape,
    num_classes,
    classifier_activation="softmax",
    preprocess=None,
):
    """Build MiniXception (see references).

    # Arguments
        input_shape: List of three integers e.g. ``[H, W, 3]``
        num_classes: Int.
        weights: ``None`` or string with pre-trained dataset. Valid datasets
            include only ``FER``.

    # Returns
        Keras model.

    # References
       - [Real-time Convolutional Neural Networks for Emotion and
            Gender Classification](https://arxiv.org/abs/1710.07557)
    """

    # base
    image_inputs = layers.Input(input_shape)
    if preprocess is None:
        x = layers.Conv2D(
            5,
            (3, 3),
            strides=(1, 1),
            use_bias=False,
        )(image_inputs)
    elif preprocess == "rescale":
        x = layers.Rescaling(scale=1 / 127.5, offset=-1)(image_inputs)
    else:
        raise ValueError("Preprocess must be None 'rescale' or 'normalize'")
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        8,
        (3, 3),
        strides=(1, 1),
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # module 1
    residual = layers.Conv2D(
        16, (1, 1), strides=(2, 2), padding="same", use_bias=False
    )(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(
        16,
        (3, 3),
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(
        16,
        (3, 3),
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = layers.add([x, residual])

    # module 2
    residual = layers.Conv2D(
        32, (1, 1), strides=(2, 2), padding="same", use_bias=False
    )(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(
        32,
        (3, 3),
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(
        32,
        (3, 3),
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = layers.add([x, residual])

    # module 3
    residual = layers.Conv2D(
        64, (1, 1), strides=(2, 2), padding="same", use_bias=False
    )(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(
        64,
        (3, 3),
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(
        64,
        (3, 3),
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = layers.add([x, residual])

    # module 4
    residual = layers.Conv2D(
        128, (1, 1), strides=(1, 1), padding="same", use_bias=False
    )(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.SeparableConv2D(
        128,
        (3, 3),
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(
        128,
        (3, 3),
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, residual])

    x = layers.Conv2D(num_classes, (3, 3), padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)  # Regularize with dropout
    output = layers.Activation(classifier_activation, name="predictions")(x)
    return keras.Model(image_inputs, output)


if __name__ == "__main__":
    import os

    os.environ["KERAS_BACKEND"] = "jax"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"

    model = MiniXception((128, 128, 3), 1)
    # model = SimpleCNN((128, 128, 3), 1)
    # model = ConvNeXtTiny((128, 128, 3), 1)
    model.summary(show_trainable=True)
