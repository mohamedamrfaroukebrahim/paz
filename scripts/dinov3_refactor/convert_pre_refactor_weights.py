"""Convert pre-refactor dinov3 .keras weights into the refactored models.

One-off. Reads each pre-refactor .keras archive's weight store directly
(zip + h5py, so the old classes are not needed), maps every stored array to
a variable of the refactored model resolved with ``get_layer``, assigns it
and saves ``<variant>_converted.weights.h5`` into a ``converted/`` folder
beside ``ported_pre_refactor/``. The original .keras files are only read.

Every source array must be used exactly once and every target variable set
exactly once with a matching shape, else the conversion stops.
"""
import argparse
import gc
import io
import os
import sys
import zipfile
from functools import partial

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import h5py  # noqa: E402
import keras  # noqa: E402

import record_reference_outputs as reference  # noqa: E402

CONVERTED_DIR = "converted"
WEIGHT_STORE = "model.weights.h5"


def read_weight_store(filepath):
    with zipfile.ZipFile(filepath, "r") as archive:
        buffer = io.BytesIO(archive.read(WEIGHT_STORE))
    arrays = {}
    with h5py.File(buffer, "r") as store:
        store.visititems(partial(collect_array, arrays))
    return arrays


def collect_array(arrays, path, item):
    if isinstance(item, h5py.Dataset):
        arrays[path] = item[()]


def indexed(name, index):
    return name if index == 0 else f"{name}_{index}"


def name_index(name):
    suffix = name.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else 0


def has_layer(model, name):
    try:
        model.get_layer(name)
    except ValueError:
        return False
    return True


def resolve(model, names):
    layer = model
    for name in names:
        layer = layer.get_layer(name)
    return layer


def pair(model, prefix, *names):
    layer = resolve(model, names)
    paths = [f"{prefix}/vars/{index}" for index in range(len(layer.weights))]
    return [(path, weight, False) for path, weight in zip(paths, layer.weights)]


def token_pairs(model):
    names = ["cls_token", "storage_tokens", "mask_token"]
    names = [name for name in names if has_layer(model, name)]
    weights = [model.get_layer(name).weights[0] for name in names]
    paths = [f"vars/{index}" for index in range(len(names))]
    return [(path, weight, True) for path, weight in zip(paths, weights)]


def vit_pairs(model, arrays):
    pairs = token_pairs(model)
    pairs += pair(model, "layers/patch_embed/projection", "patch_embed_proj")
    pairs += pair(model, "layers/rope_position_embedding", "rope_embed")
    pairs += pair(model, "layers/stable_layer_normalization", "norm")
    block_index = 0
    while has_layer(model, f"blocks_{block_index}_attn"):
        pairs += vit_block_pairs(model, block_index)
        block_index = block_index + 1
    return pairs


def vit_block_pairs(model, index):
    source = "blocks/" + indexed("self_attention_block", index)
    name = f"blocks_{index}"
    pairs = []
    for part in ("norm1", "norm2", "ls1", "ls2"):
        if has_layer(model, f"{name}_{part}"):
            pairs += pair(model, f"{source}/{part}", f"{name}_{part}")
    for part in ("qkv", "proj"):
        names = (f"{name}_attn", f"{name}_attn_{part}")
        pairs += pair(model, f"{source}/attn/{part}", *names)
    feedforward = model.get_layer(f"{name}_mlp")
    for part in ("fc1", "fc2", "w1", "w2", "w3"):
        if has_layer(feedforward, f"{name}_mlp_{part}"):
            names = (f"{name}_mlp", f"{name}_mlp_{part}")
            pairs += pair(model, f"{source}/mlp/{part}", *names)
    return pairs


def convnext_pairs(model, arrays):
    pairs = []
    for index in range(4):
        pairs += downsample_pairs(model, index)
    groups = stage_groups(arrays)
    num_stages = count_stages(model)
    if len(groups) != num_stages:
        message = f"{len(groups)} stored stages for {num_stages} model stages"
        raise ValueError(message)
    for stage, group in enumerate(groups):
        pairs += stage_pairs(model, stage, group)
    pairs += pair(model, "layers/layer_normalization", "norm")
    return pairs


def downsample_pairs(model, index):
    source = "downsample_layers/" + indexed("sequential", index)
    name = f"downsample_layers_{index}"
    norm_index, conv_index = (1, 0) if index == 0 else (0, 1)
    conv = pair(model, f"{source}/layers/conv2d", f"{name}_{conv_index}")
    norm = f"{name}_{norm_index}"
    return conv + pair(model, f"{source}/layers/layer_norm", norm)


def stage_groups(arrays):
    paths = [path.split("/") for path in arrays]
    groups = {path[1] for path in paths if path[0] == "layers"}
    groups = [group for group in groups if group.startswith("sequential")]
    return sorted(groups, key=name_index)


def count_stages(model):
    stage = 0
    while has_layer(model, f"stages_{stage}_0_dwconv"):
        stage = stage + 1
    return stage


def stage_pairs(model, stage, group):
    pairs, index = [], 0
    while has_layer(model, f"stages_{stage}_{index}_dwconv"):
        pairs += convnext_block_pairs(model, stage, group, index)
        index = index + 1
    return pairs


def convnext_block_pairs(model, stage, group, index):
    source = f"layers/{group}/layers/" + indexed("block", index)
    name = f"stages_{stage}_{index}"
    pairs = []
    for part in ("dwconv", "norm", "pwconv1", "pwconv2"):
        pairs += pair(model, f"{source}/{part}", f"{name}_{part}")
    if has_layer(model, f"{name}_gamma"):
        pairs += pair(model, source, f"{name}_gamma")
    return pairs


def match_shape(array, weight, squeeze, path):
    shape = tuple(weight.shape)
    if squeeze and array.shape == (1,) + shape:
        array = array.reshape(shape)
    if array.shape != shape:
        message = f"{path} {array.shape} does not fit {weight.path} {shape}"
        raise ValueError(message)
    return array


def assign_pairs(arrays, pairs, model):
    used_paths, set_weights = set(), set()
    for path, weight, squeeze in pairs:
        if path not in arrays:
            raise KeyError(f"no source array {path} for {weight.path}")
        if path in used_paths:
            raise ValueError(f"source array {path} used twice")
        if id(weight) in set_weights:
            raise ValueError(f"target {weight.path} set twice")
        weight.assign(match_shape(arrays[path], weight, squeeze, path))
        used_paths.add(path)
        set_weights.add(id(weight))
    check_coverage(arrays, used_paths, model, set_weights)


def check_coverage(arrays, used_paths, model, set_weights):
    unused = sorted(set(arrays) - used_paths)
    weights = model.weights
    unset = [weight.path for weight in weights if id(weight) not in set_weights]
    if len(unused) > 0 or len(unset) > 0:
        message = f"unused source arrays {unused}, unset targets {unset}"
        raise ValueError(message)


def check_not_original(source, target):
    source_dir = os.path.normcase(os.path.dirname(os.path.abspath(source)))
    target_dir = os.path.normcase(os.path.dirname(os.path.abspath(target)))
    if source_dir == target_dir or target.endswith(".keras"):
        raise ValueError(f"refusing to write {target} over the originals")


def converted_file(weights_root, directory, variant):
    folder = os.path.join(weights_root, os.path.dirname(directory))
    filename = variant + "_converted.weights.h5"
    return os.path.join(folder, CONVERTED_DIR, filename)


def convert_variant(variant, build, make_pairs, source, target):
    check_not_original(source, target)
    arrays = read_weight_store(source)
    model = build(variant)
    assign_pairs(arrays, make_pairs(model, arrays), model)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    model.save_weights(target)
    print(f"{variant}: {len(arrays)} arrays assigned, saved {target}")
    del model, arrays
    keras.backend.clear_session()
    gc.collect()


def collect_jobs(weights_root):
    jobs = []
    for variant in reference.VIT_BUILDERS:
        directory = reference.VIT_DIR
        source = reference.weights_file(weights_root, directory, variant)
        target = converted_file(weights_root, directory, variant)
        build = reference.build_vit
        jobs.append((variant, build, vit_pairs, source, target))
    for variant in reference.CONVNEXT_ARCHS:
        directory = reference.CONVNEXT_DIR
        source = reference.weights_file(weights_root, directory, variant)
        target = converted_file(weights_root, directory, variant)
        build = reference.build_convnext
        jobs.append((variant, build, convnext_pairs, source, target))
    return jobs


def convert_all(weights_root, variants):
    for job in collect_jobs(weights_root):
        variant, source = job[0], job[3]
        if len(variants) > 0 and variant not in variants:
            continue
        if not os.path.isfile(source):
            print(f"{variant}: no .keras file at {source}, skipped")
            continue
        convert_variant(*job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--weights_root", default=reference.WEIGHTS_ROOT)
    parser.add_argument("--variants", nargs="*", default=[])
    args = parser.parse_args()
    convert_all(args.weights_root, args.variants)
