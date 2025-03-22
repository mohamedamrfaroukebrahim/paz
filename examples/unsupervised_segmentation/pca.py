from sklearn.decomposition import PCA


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
