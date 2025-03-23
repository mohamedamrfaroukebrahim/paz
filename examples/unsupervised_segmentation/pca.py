import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def flatten_features(features):
    num_images, H, W, dimension = features.shape
    num_joint_features = num_images * H * W
    return features.reshape(num_joint_features, dimension)


def apply_PCA(features, dimension=3):
    num_images, H, W = features.shape[:3]
    features = flatten_features(features)
    pca = PCA(n_components=dimension)
    pca.fit(features)
    projected_features = pca.transform(features)
    return projected_features.reshape(num_images, H, W, dimension)


def apply_masked_PCA(joint_features, foreground_masks, dimension=3):
    foreground_features = mask_features(joint_features, foreground_masks)
    pca = PCA(n_components=dimension)
    pca.fit(foreground_features)
    projected_features = pca.transform(flatten_features(joint_features))
    return projected_features.reshape(*joint_features.shape[:3], dimension)


def mask_features(joint_features, foreground_masks):
    num_images, H, W, dimension = joint_features.shape
    joint_features = flatten_features(joint_features)
    masked_features = joint_features[foreground_masks.flatten()]
    return masked_features  # TODO output everything in grid image shape?


def apply_K_means(features, num_parts):
    kmeans = KMeans(n_clusters=num_parts).fit(features)
    labels = kmeans.predict(features)
    return labels


def apply_PCA_K_means(dimension, num_parts, features):
    pca = PCA(n_components=dimension)
    pca.fit(features)
    features = pca.transform(features)
    labels = apply_K_means(features, num_parts)
    return features, labels


def add_background_label(labels):
    return labels + 1


def cluster_features(cluster, num_parts, joint_features, foreground_masks):
    # TODO refactor such that apply_PCA and apply_masked_PCA can work here.
    foreground_features = mask_features(joint_features, foreground_masks)
    _, foreground_labels = cluster(num_parts, foreground_features)
    foreground_labels = add_background_label(foreground_labels)
    num_images, H, W, _ = joint_features.shape
    labels = np.zeros((num_images * H * W, 1))
    labels[foreground_masks.reshape(num_images * H * W, 1)] = foreground_labels
    num_images, H, W, _ = joint_features.shape
    return labels.reshape(num_images, H, W, 1)  # todo should we return 1?
