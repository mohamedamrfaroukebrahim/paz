import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
from keras import Model
import jax.numpy as jp

import cluttered_mnist
from stn import STN
import paz


batch_size = 256
num_epochs = 10
num_classes = 10

train_data, val_data, test_data = cluttered_mnist.load()
x_train, y_train = train_data
x_valid, y_valid = train_data
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)


model = STN((40, 40, 1))
model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
model.summary()

model.fit(
    x_train, y_train, batch_size, num_epochs, validation_data=(x_valid, y_valid)
)


attent = Model(model.input, model.get_layer("bilinear_interpolation").output)
images = attent(x_valid[:32])
true_mosaic = paz.draw.mosaic(x_valid[:32], background=1)
pred_mosaic = paz.draw.mosaic(images, background=1)
mosaic = jp.concatenate([true_mosaic, pred_mosaic], axis=1)
paz.image.show(paz.image.denormalize(mosaic))
