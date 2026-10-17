import os

os.environ["KERAS_BACKEND"] = "jax"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

HEAVY_HELP = "run heavy torch/compute tests"


def pytest_addoption(parser):
    # dinov2_legacy/conftest.py registers the same flag; tolerate a repeat
    try:
        parser.addoption("--run-heavy", action="store_true", help=HEAVY_HELP)
    except ValueError:
        pass
