import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import jax
import jax.numpy as jp
import ensemble
import paz
import deepfish
from pipeline import batch
from generator import Generator
import sklearn
import logger

models = ensemble.load("experiments/*_ensemble_*/simple.keras")
images, labels = deepfish.load("Deepfish/", "validation")
key = jax.random.PRNGKey(777)
batch_validation = jax.jit(paz.partial(batch, augment=False))
validation_generator = Generator(key, images, labels, batch_validation)

num_models = len(models)
y_data, y_model_pred = [], []
for batch_arg in tqdm(range(len(validation_generator))):
    batch_images, y_batch_true = validation_generator.__getitem__(batch_arg)
    y_data.append(y_batch_true)
    y_model_batch_pred = ensemble.predict(models, batch_images)
    y_model_pred.append(y_model_batch_pred.reshape(num_models, -1))

y_data = jp.array(y_data)
y_model_pred = jp.moveaxis(jp.array(y_model_pred), 1, 0)
y_data = y_data.reshape(-1)
y_model_pred = y_model_pred.reshape(num_models, -1)

y_data = y_data.astype(int)
accuracies, y_pred_stdvs = [], []
for arg in range(1, num_models + 1):
    y_pred_mean = jp.mean(y_model_pred[0:arg], axis=0)
    y_pred_stdv = jp.std(y_model_pred[0:arg], axis=0)
    y_pred_stdvs.append(y_pred_stdv)
    y_pred = (y_pred_mean > 0.5).astype(int)
    accuracies.append(sklearn.metrics.accuracy_score(y_data, y_pred))
accuracies = jp.array(accuracies)

root = paz.logger.make_timestamped_directory("results", "ensemble")
jp.save(os.path.join(root, "y_data.npy"), y_data)
jp.save(os.path.join(root, "y_model_pred.npy"), y_model_pred)
jp.save(os.path.join(root, "accuracies.npy"), accuracies)


def plot(y, y_range=(0.8, 0.9)):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = 20
    plt.rcParams["font.family"] = "ptm"
    plt.rcParams["font.serif"] = "phv"
    # gray = (0.662, 0.647, 0.576)
    yellow = (1.0, 0.65, 0.0)

    figure, axis = plt.subplots()
    x = range(len(y))
    axis.plot(
        x,
        y,
        marker="o",
        markersize=10,
        linewidth=3,
        color=yellow,
    )

    axis.yaxis.set_minor_locator(AutoMinorLocator())
    axis.set_ylabel("Accuracy", labelpad=20)
    axis.set_xlabel("Ensembles", labelpad=20)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    # axis.fill_between(x, y - y_error, y + y_error, color=yellow, alpha=0.25)
    # axis.set_xlim(x_range)
    axis.set_ylim(y_range)
    plt.show()
    # figure.savefig(filepath, bbox_inches="tight")
    # plt.close()
