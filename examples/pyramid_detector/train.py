# TODO make plot of augmentations of a single image
import os
import argparse

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"

import jax
import paz
import keras
from keras.optimizers import Adam, AdamW

from deepfish import load
from pipeline import batch
from generator import Generator
from models import FineTuneXception, SimpleCNN, MiniXception, ConvNeXtTiny
from vit2 import ViT
import plot

parser = argparse.ArgumentParser(description="Train fish classifier")
MODELS = ["simple", "minixception", "xception", "convnext", "vit"]
parser.add_argument("--seed", default=777, type=int)
parser.add_argument("--root", default="experiments", type=str)
parser.add_argument("--label", default=None)
parser.add_argument("--model", default="simple", type=str, choices=MODELS)
parser.add_argument("--box_H", default=128, type=int)
parser.add_argument("--box_W", default=128, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--max_epochs", default=100, type=int)
parser.add_argument("--optimizer", default="adamw", choices=["adam", "adamw"])
parser.add_argument("--weight_decay", default=1e-3, type=float)
parser.add_argument("--learning_rate", default=5e-4, type=float)
parser.add_argument("--stop_patience", default=8, type=int)
parser.add_argument("--scale_patience", default=4, type=int)
args = parser.parse_args()
key = jax.random.PRNGKey(args.seed)
keras.utils.set_random_seed(args.seed)

labels = [args.model] if args.label is None else [args.model, args.label]
root = paz.logger.make_timestamped_directory(args.root, "_".join(labels))
paz.logger.write_dictionary(args.__dict__, root, "hyper-parameters.json")

train_images, train_labels = load("Deepfish/", "train")
valid_images, valid_labels = load("Deepfish/", "validation")
batch_train = jax.jit(batch)
batch_valid = jax.jit(paz.partial(batch, augment=False))
train_generator = Generator(key, train_images, train_labels, batch_train)
valid_generator = Generator(key, valid_images, valid_labels, batch_valid)

batch_images = train_generator.__getitem__(0)[0]
paz.image.write(
    os.path.join(root, "train_batch.png"),
    paz.draw.mosaic(batch_images.astype("uint8"), border=5).astype("uint8"),
)


Model = {
    "xception": FineTuneXception,
    "convnext": ConvNeXtTiny,
    "simple": SimpleCNN,
    "minixception": paz.partial(
        MiniXception, classifier_activation="linear", preprocess="rescale"
    ),
    "vit": paz.lock(ViT, 8, 16, 2, [32, 16], 2, [8, 8]),
}
model = Model[args.model]((args.box_H, args.box_W, 3), 1)
model.summary(show_trainable=True)
keras.utils.plot_model(
    model,
    os.path.join(root, args.model + ".png"),
    show_shapes=True,
    show_trainable=True,
)

if args.optimizer == "adam":
    optimizer = Adam(args.learning_rate, weight_decay=args.weight_decay)
elif args.optimizer == "adamw":
    optimizer = AdamW(args.learning_rate, weight_decay=args.weight_decay)
else:
    raise ValueError(f"Optimizer {args.optimizer} not supported")

model.compile(
    optimizer=optimizer,
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
    jit_compile=True,
)

callbacks = [
    keras.callbacks.EarlyStopping(patience=args.stop_patience),
    keras.callbacks.ReduceLROnPlateau(patience=args.scale_patience),
    keras.callbacks.CSVLogger(os.path.join(root, "log.csv")),
    keras.callbacks.ModelCheckpoint(
        os.path.join(root, f"{args.model}.keras"), save_best_only=True
    ),
]

fit = model.fit(
    train_generator,
    batch_size=args.batch_size,
    epochs=args.max_epochs,
    validation_data=valid_generator,
    callbacks=callbacks,
)


plot.accuracies(
    [fit.history["val_binary_accuracy"], fit.history["binary_accuracy"]],
    ["Validation", "Train"],
    os.path.join(root, "accuracy.pdf"),
)


plot.binary_cross_entropies(
    [fit.history["val_loss"], fit.history["loss"]],
    ["Validation", "Train"],
    os.path.join(root, "losses.pdf"),
)
