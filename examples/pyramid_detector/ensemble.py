import os

os.environ["KERAS_BACKEND"] = "jax"
import matplotlib.pyplot as plt
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
    entropy = compute_entropy(predictions, num_bins)
    plt.hist(predictions, bins=num_bins)
    plt.title(f"Entropy: {entropy:.4f}")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
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
    sample_arg = 0
    image = batch_images[sample_arg]
    label = batch_labels[sample_arg]
    paz.image.show(image)
    predictions = predict(models, image[None, ...])[:, 0, 0]
    print("Entropy:", compute_entropy(predictions, num_bins=10))
    print("Label:", label, "|", "Mean Pred:", jp.mean(predictions))
    plot_entropy(predictions, num_bins=10)
