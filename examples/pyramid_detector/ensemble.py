import os

os.environ["KERAS_BACKEND"] = "jax"
import matplotlib.pyplot as plt
import matplotlib
from glob import glob
import jax
import jax.numpy as jp
import keras


def load(wildcard):
    return [keras.saving.load_model(filepath) for filepath in glob(wildcard)]


def predict(models, image, activation=jax.nn.sigmoid):
    return jp.array([activation(model(image)) for model in models])


def compute_entropy(predictions, num_bins=10):
    """Computes the entropy of the empirical distribution of an ensemble's
    probability predictions using histogram binning.
    """

    def p_log2_p(p, epsilon=1e-12):
        log_p = jp.log2(p + epsilon)
        return p * log_p

    counts, _ = jp.histogram(predictions, bins=num_bins, range=(0.0, 1.0))
    probability_mass = counts / jp.sum(counts)
    entropy = -jp.sum(p_log2_p(probability_mass))

    return entropy


def plot_entropy(predictions, num_bins=10):
    yellow = (1.0, 0.65, 0.0)
    gray = (0.662, 0.647, 0.576)
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = 20
    plt.rcParams["font.family"] = "ptm"
    plt.rcParams["font.serif"] = "phv"

    entropy = compute_entropy(predictions, num_bins)
    figure, axis = plt.subplots()
    axis.set_ylabel("Accuracy")
    axis.set_xlabel("Epochs")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    mean = jp.mean(predictions)
    axis.axvline(mean, color=gray, linestyle="--", label="Mean", ymax=0.7)
    y_text = len(predictions) * 0.5
    axis.set_xlim([0, 1])
    axis.text(mean - 0.1, y_text, "Mean: {:.2f}".format(mean), color=gray)

    axis.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    axis.xaxis.labelpad = 20
    axis.yaxis.labelpad = 20

    axis.hist(predictions, bins=num_bins, color=yellow)
    axis.set_title(f"Entropy: {entropy:.4f}")
    axis.set_xlabel("Posterior Probability")
    axis.set_ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    import paz
    import deepfish
    from pipeline import batch
    from generator import Generator

    models = load("experiments/*_ensemble_*/simple.keras")
    images, labels = deepfish.load("Deepfish/", "validation")
    key = jax.random.PRNGKey(777)
    batch_valid = jax.jit(paz.partial(batch, augment=False))
    train_generator = Generator(key, images, labels, batch_valid)
    batch_images, batch_labels = train_generator.__getitem__(0)
    sample_arg = 2
    image = batch_images[sample_arg]
    label = batch_labels[sample_arg]
    paz.image.show(image)
    predictions = predict(models, image[None, ...])[:, 0, 0]
    print("Entropy:", compute_entropy(predictions, num_bins=10))
    print("Label:", label, "|", "Mean Pred:", jp.mean(predictions))
    plot_entropy(predictions, num_bins=10)
