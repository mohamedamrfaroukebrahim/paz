from itertools import product
import jax.numpy as jp


def create_prior_boxes(configuration_name="VOC"):
    configuration = get_prior_box_configuration(configuration_name)
    image_size = configuration["image_size"]
    feature_map_sizes = configuration["feature_map_sizes"]
    min_sizes = configuration["min_sizes"]
    max_sizes = configuration["max_sizes"]
    steps = configuration["steps"]
    model_aspect_ratios = configuration["aspect_ratios"]
    mean = []
    for feature_map_arg, feature_map_size in enumerate(feature_map_sizes):
        step = steps[feature_map_arg]
        min_size = min_sizes[feature_map_arg]
        max_size = max_sizes[feature_map_arg]
        aspect_ratios = model_aspect_ratios[feature_map_arg]
        for y, x in product(range(feature_map_size), repeat=2):
            f_k = image_size / step
            center_x = (x + 0.5) / f_k
            center_y = (y + 0.5) / f_k
            s_k = min_size / image_size
            mean = mean + [center_x, center_y, s_k, s_k]
            s_k_prime = jp.sqrt(s_k * (max_size / image_size))
            mean = mean + [center_x, center_y, s_k_prime, s_k_prime]
            for aspect_ratio in aspect_ratios:
                mean = mean + [
                    center_x,
                    center_y,
                    s_k * jp.sqrt(aspect_ratio),
                    s_k / jp.sqrt(aspect_ratio),
                ]
                mean = mean + [
                    center_x,
                    center_y,
                    s_k / jp.sqrt(aspect_ratio),
                    s_k * jp.sqrt(aspect_ratio),
                ]

    output = jp.asarray(mean).reshape((-1, 4))
    return output


def get_prior_box_configuration(configuration_name="VOC"):
    if configuration_name in {"VOC", "FAT"}:
        configuration = {
            "feature_map_sizes": [38, 19, 10, 5, 3, 1],
            "image_size": 300,
            "steps": [8, 16, 32, 64, 100, 300],
            "min_sizes": [30, 60, 111, 162, 213, 264],
            "max_sizes": [60, 111, 162, 213, 264, 315],
            "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            "variance": [0.1, 0.2],
        }

    elif configuration_name in {"COCO", "YCBVideo"}:
        configuration = {
            "feature_map_sizes": [64, 32, 16, 8, 4, 2, 1],
            "image_size": 512,
            "steps": [8, 16, 32, 64, 128, 256, 512],
            "min_sizes": [21, 51, 133, 215, 297, 379, 461],
            "max_sizes": [51, 133, 215, 297, 379, 461, 542],
            "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
            "variance": [0.1, 0.2],
        }
    else:
        raise ValueError("Invalid configuration name:", configuration_name)
    return configuration
