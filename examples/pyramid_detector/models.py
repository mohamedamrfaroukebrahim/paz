import keras


def simpleCNN(shape=(128, 128, 3)):
    return keras.Sequential(
        [
            keras.Input(shape=shape),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )


def FineTuneXception(shape=(128, 128, 3)):
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
    outputs = keras.layers.Dense(1)(x)
    return keras.Model(inputs, outputs)
