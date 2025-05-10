import os

os.environ["KERAS_BACKEND"] = "jax"
from glob import glob
import jax
import jax.numpy as jp
import keras


def load(wildcard):
    return [keras.saving.load_model(filepath) for filepath in glob(wildcard)]


def predict(models, image, activation=jax.nn.sigmoid):
    return jp.array([activation(jp.squeeze(model(image))) for model in models])


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


if __name__ == "__main__":
    import deepfish
    from pipeline import batch
    from generator import Generator

    models = load("experiments/*_ensemble_*/simple.keras")
    train_images, train_labels = deepfish.load("Deepfish/", "train")
    key = jax.random.PRNGKey(777)
    train_generator = Generator(key, train_images, train_labels, batch)
    image = train_generator.__getitem__(0)[0][0:1]
    predictions = predict(models, image)
