from glob import glob
import jax
import jax.numpy as jp
import keras


def load(wildcard):
    return [keras.saving.load_model(filepath) for filepath in glob(wildcard)]


def predict(models, image, activation=jax.nn.sigmoid):
    return jp.array([activation(model.predict(image)) for model in models])


def compute_entropy(predictions, num_bins=10):
    """Computes the entropy of the empirical distribution of an ensemble's
    probability predictions using histogram binning.
    """

    def p_log_p(p, epsilon=1e-12):
        log_p = jp.log2(p + epsilon)
        return p * log_p

    counts, _ = jp.histogram(predictions, bins=num_bins, range=(0.0, 1.0))
    probability_mass = counts / jp.sum(counts)
    # remove vmap
    entropy = -jp.sum(jax.vmap(p_log_p)(probability_mass))
    return entropy
