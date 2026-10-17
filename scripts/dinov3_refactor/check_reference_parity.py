"""Check converted refactored dinov3 models against the recorded reference.

Builds each refactored ViT and ConvNeXt variant, loads the weights written
by convert_pre_refactor_weights.py, runs the seeded input stored in
test_outputs/dinov3_reference_outputs.npz (``image`` for the ViT,
``convnext_image`` for ConvNeXt) and asserts that the model's
``(class_token, patch_tokens)`` outputs match the recorded pre-refactor
``<variant>.class_token`` and ``<variant>.patch_tokens``. Prints the max
absolute difference per variant, then fails if any variant mismatches or
has no reference or converted weights. The npz is only read.

Tolerances are the model-level values of the dinov3 test files:
  loading_test.py:50                      rtol=1e-5, atol=1e-5
  models/vision_transformer_test.py:201   rtol=1e-5, atol=1e-5
  weight_porting_test.py:298              rtol=1e-5, atol=1e-5
  dinov3_test.py:244                      rtol=1e-5, atol=1e-5
loading_test.py:50 is the closest match: the same model's outputs before
and after its weights are reloaded, compared pair by pair.
"""
import argparse
import gc
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import keras  # noqa: E402
import numpy as np  # noqa: E402

import convert_pre_refactor_weights as converter  # noqa: E402
import record_reference_outputs as reference  # noqa: E402

RTOL = 1e-5  # loading_test.py:50, models/vision_transformer_test.py:201
ATOL = 1e-5  # loading_test.py:50, models/vision_transformer_test.py:201
OUTPUT_NAMES = ("class_token", "patch_tokens")
HEADER = ("variant", "max|d| class", "max|d| patch", "status")
ROW_FORMAT = "{:<24}{:>14}{:>14}  {}"


def collect_jobs(weights_root):
    jobs = []
    for variant in reference.VIT_BUILDERS:
        directory = reference.VIT_DIR
        filepath = converter.converted_file(weights_root, directory, variant)
        jobs.append((variant, reference.build_vit, "image", filepath))
    for variant in reference.CONVNEXT_ARCHS:
        directory = reference.CONVNEXT_DIR
        filepath = converter.converted_file(weights_root, directory, variant)
        build = reference.build_convnext
        jobs.append((variant, build, "convnext_image", filepath))
    return jobs


def missing_reason(recorded, variant, filepath):
    keys = [f"{variant}.{name}" for name in OUTPUT_NAMES]
    if not all(key in recorded.files for key in keys):
        return "no recorded reference"
    if not os.path.isfile(filepath):
        return f"no converted weights at {filepath}"
    return None


def run_variant(variant, build, image, filepath):
    model = build(variant)
    model.load_weights(filepath)
    outputs = model(image, training=False)
    outputs = [reference.to_numpy(output) for output in outputs]
    del model
    keras.backend.clear_session()
    gc.collect()
    return dict(zip(OUTPUT_NAMES, outputs))


def max_difference(value, expected):
    if value.shape != expected.shape:
        return "shape"
    difference = np.abs(value.astype("float64") - expected)
    return f"{np.max(difference):.3e}"


def assert_close(variant, name, value, expected):
    message = f"{variant} {name} differs from the pre-refactor reference"
    tolerances = dict(rtol=RTOL, atol=ATOL, err_msg=message)
    try:
        np.testing.assert_allclose(value, expected, **tolerances)
    except AssertionError as error:
        return [str(error)]
    return []


def compare(variant, outputs, recorded):
    row, failures = [variant], []
    for name in OUTPUT_NAMES:
        value, expected = outputs[name], recorded[f"{variant}.{name}"]
        row.append(max_difference(value, expected))
        failures += assert_close(variant, name, value, expected)
    row.append("PASS" if len(failures) == 0 else "FAIL")
    return row, failures


def check_variant(job, recorded):
    variant, build, image_key, filepath = job
    reason = missing_reason(recorded, variant, filepath)
    if reason is not None:
        return [variant, "-", "-", "MISSING"], [f"{variant}: {reason}"]
    outputs = run_variant(variant, build, recorded[image_key], filepath)
    return compare(variant, outputs, recorded)


def print_table(rows):
    print(f"tolerances: rtol={RTOL}, atol={ATOL}")
    print(ROW_FORMAT.format(*HEADER))
    for row in rows:
        print(ROW_FORMAT.format(*row))


def check_all(weights_root, reference_path, variants):
    recorded = np.load(reference_path)
    print(f"reference {reference_path}, seed {int(recorded['seed'])}")
    rows, failures = [], []
    for job in collect_jobs(weights_root):
        if len(variants) > 0 and job[0] not in variants:
            continue
        row, variant_failures = check_variant(job, recorded)
        print(ROW_FORMAT.format(*row))
        rows.append(row)
        failures += variant_failures
    print_table(rows)
    if len(rows) == 0:
        raise AssertionError(f"no variant selected from {variants}")
    if len(failures) > 0:
        raise AssertionError("\n".join(failures))
    print(f"all {len(rows)} variants match the pre-refactor reference")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--weights_root", default=reference.WEIGHTS_ROOT)
    parser.add_argument("--reference_path", default=reference.OUTPUT_PATH)
    parser.add_argument("--variants", nargs="*", default=[])
    args = parser.parse_args()
    check_all(args.weights_root, args.reference_path, args.variants)
