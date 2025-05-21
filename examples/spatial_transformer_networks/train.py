import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras import Model

import cluttered_mnist

import paz


batch_size = 256
num_epochs = 20
num_classes = 10

train_data, val_data, test_data = cluttered_mnist.load()
x_train, y_train = train_data
x_valid, y_valid = train_data
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

model = paz.models.STN((40, 40, 1), (28, 28), num_classes)
model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
model.summary()

model.fit(
    x_train, y_train, batch_size, num_epochs, validation_data=(x_valid, y_valid)
)


attent = Model(model.input, model.get_layer("bilinear_interpolation").output)
to_se2 = Model(model.input, model.get_layer("to_se2").output)

images = attent(x_valid[:32])
matrix = to_se2(x_valid[:32])
print(matrix)
true_mosaic = paz.draw.mosaic(x_valid[:32], background=1)
paz.image.show(paz.image.denormalize(true_mosaic))

pred_mosaic = paz.draw.mosaic(images, background=1)
paz.image.show(paz.image.denormalize(pred_mosaic))
