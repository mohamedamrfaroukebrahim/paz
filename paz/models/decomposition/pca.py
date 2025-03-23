import jax.numpy as jp
import paz


def compute_mean(x):
    return jp.mean(x, axis=0)


def subtract_mean(x, mean):
    return x - mean


def compute_feature_covariance_matrix(x):
    return jp.cov(x.T)


def sort_in_descending_order(eigenvalues, eigenvectors):
    return eigenvalues[::-1], eigenvectors[::-1]


def compute_eigenvectors(covariance_matrix):
    eigenvalues, eigenvectors = jp.linalg.eigh(covariance_matrix)
    return sort_in_descending_order(eigenvalues, eigenvectors.T)


def trim_components(eigenvalues, eigenvectors, num_components):
    return eigenvalues[:num_components], eigenvectors[:num_components]


def fit(x, num_components, total_variance=None):
    mean = compute_mean(x)
    x = subtract_mean(x, mean)
    covariance_matrix = compute_feature_covariance_matrix(x)
    values, vectors = compute_eigenvectors(covariance_matrix)
    values, vectors = trim_components(values, vectors, num_components)
    return paz.NamedTuple("PCAState", mean=mean, variances=values, base=vectors)


def transform(x, state, sample_axis=0):
    x = subtract_mean(x, state.mean)
    return (state.base @ x.T).T
