import os
import argparse

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"

import jax
import paz
import keras

from deepfish import load
from pipeline import batch
from generator import Generator
from models import FineTuneXception


parser = argparse.ArgumentParser(description="Train fish classifier")
parser.add_argument("--seed", default=777, type=int)
parser.add_argument("--root", default="experiments", type=str)
parser.add_argument("--model", default="xception", type=str)
parser.add_argument("--box_H", default=128, type=int)
parser.add_argument("--box_W", default=128, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--max_epochs", default=100, type=int)
parser.add_argument("--learning_rate", default=1e-3, type=float)
parser.add_argument("--optimizer", default="adamw", choices=["adam", "adamw"])
parser.add_argument("--stop_patience", default=5, type=int)
parser.add_argument("--scale_patience", default=3, type=int)
args = parser.parse_args()
key = jax.random.PRNGKey(args.seed)

root = paz.logger.make_timestamped_directory(args.root)
paz.logger.write_dictionary(args.__dict__, root, "hyper-parameters.json")


train_images, train_labels = load("Deepfish/", "train")
valid_images, valid_labels = load("Deepfish/", "validation")
batch_train = jax.jit(batch)
batch_valid = jax.jit(paz.partial(batch, augment=False))
train_generator = Generator(key, train_images, train_labels, batch_train)
valid_generator = Generator(key, valid_images, valid_labels, batch_valid)

model = FineTuneXception((args.box_H, args.box_W, 3))
model.summary(show_trainable=True)
keras.utils.plot_model(
    model, os.path.join(root, args.model), show_shapes=True, show_trainable=True
)

Optimizer = {"adam": keras.optimizers.Adam, "adamw": keras.optimizers.AdamW}
optimizer = Optimizer[args.optimizer](learning_rate=args.learning_rate)

model.compile(
    optimizer=optimizer,
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
    jit_compile=True,
)

callbacks = [
    keras.callbacks.EarlyStopping(patience=args.stop_patience),
    keras.callbacks.ModelCheckpoint(f"{args.model}.h5", save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(patience=args.scale_patience),
    keras.callbacks.CSVLogger(os.path.join(root, "log.csv")),
]

model.fit(
    train_generator,
    batch_size=args.batch_size,
    epochs=args.max_epochs,
    validation_data=valid_generator,
    callbacks=callbacks,
)
