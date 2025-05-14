import keras


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


if __name__ == "__main__":
    import os

    os.environ["KERAS_BACKEND"] = "jax"
    import jax.numpy as jp

    # model = SimpleCNN((128, 128, 3), 1)
    from paz.models.classification import MiniXception

    model = MiniXception(input_shape=(128, 128, 3), num_classes=1)
    logits = model(jp.zeros((10, 128, 128, 3)))
    print(logits.shape)
