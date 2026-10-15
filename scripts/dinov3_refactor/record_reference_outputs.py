"""Record reference outputs of the pre-refactor dinov3 models.

Run on the PRE-REFACTOR tree, before any dinov3 source file changes. Builds
each ViT and ConvNeXt variant exactly as the dinov3 porter does, loads its
ported .keras weights, runs one seeded image batch and saves the inputs, the
raw outputs and a weight path/shape listing per variant to
test_outputs/dinov3_reference_outputs.npz.

Keys in the npz:
  seed, image (NHWC, ViT input), convnext_image (NCHW, ConvNeXt input),
  variants, missing_variants, and per recorded variant `<variant>.<field>`:
  output (model call), class_token, patch_tokens, prenorm,
  storage_tokens (ViT only), weights_file, weight_paths, weight_shapes.
"""
import argparse
import gc
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import keras  # noqa: E402
import numpy as np  # noqa: E402

from paz.models.foundation.dinov3.models import convnext  # noqa: E402
from paz.models.foundation.dinov3.models import vision_transformer  # noqa: E402

vit = vision_transformer

WEIGHTS_ROOT = "D:/DFKI_SeaMe_project/Tasks/Task2_porting_paz_model_to_keras3"
VIT_DIR = os.path.join("dinov3 weights", "ported_pre_refactor")
CONVNEXT_DIR = os.path.join("dinov3 convnext weights", "ported_pre_refactor")
OUTPUT_PATH = os.path.join(ROOT, "test_outputs", "dinov3_reference_outputs.npz")

VIT_BUILDERS = {
    "dinov3_vits16": vit.vit_small,
    "dinov3_vitb16": vit.vit_base,
    "dinov3_vitl16": vit.vit_large,
}

# same ViT kwargs as port_dino_weights_from_torch_to_keras.py
VIT_KWARGS = {
    "img_size": 224,
    "patch_size": 16,
    "ffn_layer": "mlp",
    "untie_cls_and_patch_norms": False,
    "norm_layer": "layernorm",
    "layerscale_init": 1e-6,
    "n_storage_tokens": 4,
    "pos_embed_rope_dtype": "float32",
}

CONVNEXT_ARCHS = {
    "dinov3_convnext_tiny": "convnext_tiny",
    "dinov3_convnext_small": "convnext_small",
    "dinov3_convnext_base": "convnext_base",
    "dinov3_convnext_large": "convnext_large",
}

# same ConvNeXt kwargs as the porter (img_size is only its dummy input size)
CONVNEXT_KWARGS = {"patch_size": 16, "layer_scale_init_value": 1e-6}


def make_image(seed, image_size):
    random_state = np.random.default_rng(seed)
    shape = (1, image_size, image_size, 3)
    return random_state.standard_normal(shape).astype("float32")


def to_numpy(x):
    return np.asarray(keras.ops.convert_to_numpy(x))


def weights_file(weights_root, directory, variant):
    return os.path.join(weights_root, directory, variant + "_ported.keras")


def build_vit(variant):
    model = VIT_BUILDERS[variant](**VIT_KWARGS)
    image_size = VIT_KWARGS["img_size"]
    model(np.zeros((1, image_size, image_size, 3), "float32"), training=False)
    return model


def build_convnext(variant):
    builder = convnext.get_convnext_arch(CONVNEXT_ARCHS[variant])
    model = builder(**CONVNEXT_KWARGS)
    model(np.zeros((1, 3, 224, 224), "float32"), training=False)
    return model


def list_weights(model):
    paths = [weight.path for weight in model.weights]
    shapes = [str(tuple(weight.shape)) for weight in model.weights]
    return np.array(paths), np.array(shapes)


def run_vit(model, image):
    features = model.forward_features(image, training=False)
    outputs = {"output": model(image, training=False)}
    outputs["class_token"] = features["x_norm_clstoken"]
    outputs["storage_tokens"] = features["x_storage_tokens"]
    outputs["patch_tokens"] = features["x_norm_patchtokens"]
    outputs["prenorm"] = features["x_prenorm"]
    return outputs


def run_convnext(model, image):
    features = model.forward_features(image)
    outputs = {"output": model(image, training=False)}
    outputs["class_token"] = features["x_norm_clstoken"]
    outputs["patch_tokens"] = features["x_norm_patchtokens"]
    outputs["prenorm"] = features["x_prenorm"]
    return outputs


def record_variant(variant, build, run, filepath, image):
    model = build(variant)
    model.load_weights(filepath)
    outputs = run(model, image)
    record = {key: to_numpy(value) for key, value in outputs.items()}
    weight_paths, weight_shapes = list_weights(model)
    record["weights_file"] = np.array(filepath)
    record["weight_paths"] = weight_paths
    record["weight_shapes"] = weight_shapes
    del model
    keras.backend.clear_session()
    gc.collect()
    return record


def describe(variant, record):
    shapes = {key: value.shape for key, value in record.items()}
    num_weights = len(record["weight_paths"])
    print("{}: {} weights, {}".format(variant, num_weights, shapes))


def collect_jobs(weights_root, image, convnext_image):
    jobs = []
    for variant in VIT_BUILDERS:
        filepath = weights_file(weights_root, VIT_DIR, variant)
        jobs.append((variant, build_vit, run_vit, filepath, image))
    for variant in CONVNEXT_ARCHS:
        filepath = weights_file(weights_root, CONVNEXT_DIR, variant)
        job = (variant, build_convnext, run_convnext, filepath, convnext_image)
        jobs.append(job)
    return jobs


def record_all(weights_root, seed, output_path):
    image = make_image(seed, VIT_KWARGS["img_size"])
    convnext_image = np.transpose(image, (0, 3, 1, 2))
    arrays = {"seed": np.array(seed), "image": image}
    arrays["convnext_image"] = convnext_image
    recorded, missing = [], []
    for job in collect_jobs(weights_root, image, convnext_image):
        variant, filepath = job[0], job[3]
        if not os.path.isfile(filepath):
            print("{}: no .keras file at {}, skipped".format(variant, filepath))
            missing.append(variant)
            continue
        record = record_variant(*job)
        describe(variant, record)
        for key, value in record.items():
            arrays[variant + "." + key] = value
        recorded.append(variant)
    arrays["variants"] = np.array(recorded)
    arrays["missing_variants"] = np.array(missing, dtype=str)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, **arrays)
    print("saved {} variants to {}".format(len(recorded), output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--weights_root", default=WEIGHTS_ROOT)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--output_path", default=OUTPUT_PATH)
    args = parser.parse_args()
    record_all(args.weights_root, args.seed, args.output_path)
