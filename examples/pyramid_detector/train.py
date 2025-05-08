import os

os.environ["KERAS_BACKEND"] = "jax"
import jax
import jax.numpy as jp
import paz
from paz.models.classification import MiniXception
import keras

from deepfish import load
from pipeline import batch
from generator import Generator

train_images, train_labels = load("Deepfish/", "train")
valid_images, valid_labels = load("Deepfish/", "validation")
box_size = (128, 128)
batch_size = 32
epochs = 10
key = jax.random.PRNGKey(777)

generator = Generator(key, jax.jit(batch), train_images, train_labels)
batch_images, batch_labels = generator.__getitem__(0)
paz.image.show(paz.draw.mosaic(batch_images.astype("uint8")).astype("uint8"))


# model = keras.Sequential(
#     [
#         keras.Input(shape=(128, 128, 3)),
#         keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
#         keras.layers.MaxPooling2D(pool_size=(2, 2)),
#         keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#         keras.layers.MaxPooling2D(pool_size=(2, 2)),
#         keras.layers.Flatten(),
#         keras.layers.Dense(1, activation="sigmoid"),
#     ]
# )

base_model = keras.applications.Xception(
    weights="imagenet",
    input_shape=(128, 128, 3),
    include_top=False,
)

base_model.trainable = False

inputs = keras.Input(shape=(128, 128, 3))
scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(inputs)

x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.summary(show_trainable=True)

model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=[keras.metrics.BinaryAccuracy()],
)

model.fit(generator, batch_size=batch_size, epochs=epochs)
#
