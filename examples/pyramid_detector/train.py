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
from models import FineTuneXception

train_images, train_labels = load("Deepfish/", "train")
valid_images, valid_labels = load("Deepfish/", "validation")
box_size = (128, 128)
batch_size = 32
epochs = 10
key = jax.random.PRNGKey(777)

batch_train = jax.jit(batch)
batch_valid = jax.jit(paz.partial(batch, augment=False))
train_generator = Generator(key, train_images, train_labels, batch_train)
valid_generator = Generator(key, valid_images, valid_labels, batch_valid)

batch_images, batch_labels = train_generator.__getitem__(0)
paz.image.show(paz.draw.mosaic(batch_images.astype("uint8")).astype("uint8"))

model = FineTuneXception((*box_size, 3))
model.summary(show_trainable=True)

model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=[keras.metrics.BinaryAccuracy()],
    jit_compile=True,
)

model.fit(
    train_generator,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=valid_generator,
)
