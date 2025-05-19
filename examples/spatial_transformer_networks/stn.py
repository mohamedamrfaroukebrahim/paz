import os

os.environ["KERAS_BACKEND"] = "jax"
import numpy as np
import keras
from keras.layers import Input, Activation, MaxPool2D, Flatten, Layer
from keras.layers import Conv2D, Dense, Lambda
import jax
import paz


def transform(images, matrices):
    return jax.vmap(paz.image.affine_transform)(images, matrices)


class BilinearInterpolation(Layer):
    def __init__(self):
        super().__init__()

    def call(self, images_and_transforms):
        images, affine_transforms = images_and_transforms
        return transform(images, affine_transforms)


class ToSE2(Layer):
    def __init__(self):
        super().__init__()

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 3, 3)

    def call(self, rotation_and_translation):
        rotation, translation = rotation_and_translation
        rotation = rotation.reshape((-1, 2, 2))
        return jax.vmap(paz.SE2.to_affine_matrix)(rotation, translation)


def initialize_interpolation_weights(output_size):
    b = np.zeros((2, 3), dtype="float32")
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((output_size, 6), dtype="float32")
    weights = [W, b.flatten()]
    return weights


def STN(input_shape=(40, 40, 1), interpolation_size=(30, 30), num_classes=10):
    image = Input(shape=input_shape)
    x = MaxPool2D(pool_size=(2, 2))(image)
    x = Conv2D(20, (5, 5))(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(20, (5, 5))(x)
    x = Flatten()(x)
    x = Dense(50)(x)
    x = Activation("relu")(x)
    r = Dense(4, kernel_initializer="zeros", bias_initializer="ones")(x)
    t = Dense(2, kernel_initializer="zeros", bias_initializer="ones")(x)
    a = ToSE2()([r, t])
    interpolated_image = BilinearInterpolation()([image, a])
    x = Conv2D(32, (3, 3), padding="same")(interpolated_image)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (3, 3))(x)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation("relu")(x)
    x = Dense(num_classes)(x)
    x = Activation("softmax", name="label")(x)
    return keras.Model(image, x, name="STN")


if __name__ == "__main__":
    STN().summary()
