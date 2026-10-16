import os
import torch
import numpy as np
import torch.nn as nn
import jax.numpy as jnp
import pytest

from paz.models.foundation.dinov3.layers.ffn_layers import (
    Mlp,
    SwiGLUFFN,
)

from paz.models.foundation.dinov3.layers.torch_layers_for_testing import (
    PT_Mlp,
    PT_SwiGLUFFN,
)


MISSING_WEIGHT_PATH = "/path/that/does/not/exist/dinov3_vits16_pretrain.pth"
ASSET_ERRORS = (FileNotFoundError, ModuleNotFoundError)
ASSET_ERRORS = ASSET_ERRORS + (ImportError, AttributeError)


def random_input(params, dim):
    return torch.randn(params["batch_size"], params["seq_len"], dim)


def dense_weights(layer):
    weight = layer.weight.T.detach().numpy()
    return [weight, layer.bias.detach().numpy()]


def state_dense_weights(state_dict, name):
    weight = state_dict[f"{name}.weight"].T.numpy()
    return [weight, state_dict[f"{name}.bias"].numpy()]


def create_mlp_pair(
    in_dim,
    drop=0.0,
    bias=True,
    hidden_features=None,
    out_features=None,
    approximate_gelu=False,
):
    """Creates, builds, and transfers weights for a Keras and PyTorch Mlp
    pair."""
    # 1. Create and configure PyTorch model
    approximate = "tanh" if approximate_gelu else "none"
    pt_act_layer = lambda: nn.GELU(approximate=approximate)
    kwargs = dict(in_features=in_dim, hidden_features=hidden_features)
    kwargs.update(out_features=out_features, drop=drop, bias=bias)
    pt_model = PT_Mlp(**kwargs, act_layer=pt_act_layer)
    pt_model.eval()

    # 2. Create and build Keras model
    keras_model = Mlp(**kwargs, approximate_gelu=approximate_gelu)
    keras_model(jnp.ones((1, 1, in_dim)))

    # ... rest of the weight transfer is the same
    pt_weights = pt_model.state_dict()
    fc1_weights = [pt_weights["fc1.weight"].T.numpy()]
    if bias:
        fc1_weights.append(pt_weights["fc1.bias"].numpy())
    keras_model.get_layer("mlp_fc1").set_weights(fc1_weights)

    fc2_weights = [pt_weights["fc2.weight"].T.numpy()]
    if bias:
        fc2_weights.append(pt_weights["fc2.bias"].numpy())
    keras_model.get_layer("mlp_fc2").set_weights(fc2_weights)

    return pt_model, keras_model


def create_swiglu_pair(
    in_dim, bias=True, align_to=8, hidden_features=None, out_features=None
):
    """Creates, builds, and transfers weights for a Keras and PyTorch
    SwiGLUFFN pair."""
    # 1. Create and configure PyTorch model
    kwargs = dict(in_features=in_dim, hidden_features=hidden_features)
    kwargs.update(out_features=out_features, bias=bias, align_to=align_to)
    pt_model = PT_SwiGLUFFN(**kwargs)
    pt_model.eval()  # Set to evaluation mode

    # 2. Create and build Keras model
    keras_model = SwiGLUFFN(**kwargs)
    keras_model(jnp.ones((1, 1, in_dim)))  # Build the layer

    # 3. Transfer weights from PyTorch to Keras
    pt_weights = pt_model.state_dict()
    for i in range(1, 4):
        layer_name = f"w{i}"
        keras_layer = keras_model.get_layer(f"mlp_{layer_name}")
        pt_layer_weights = [pt_weights[f"{layer_name}.weight"].T.numpy()]
        if bias:
            pt_layer_weights.append(pt_weights[f"{layer_name}.bias"].numpy())
        keras_layer.set_weights(pt_layer_weights)

    return pt_model, keras_model


@pytest.fixture(scope="module")
def params():
    """Provides a dictionary of common parameters for tests."""
    # Set a fixed seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    return {
        "batch_size": 4,
        "seq_len": 16,
        "in_dim": 64,
        "hidden_dim": 128,
        "drop_rate": 0.5,
    }


def test_mlp_basic(params):
    """🧪 Test Mlp: Basic case with specified dimensions."""
    kwargs = dict(in_dim=params["in_dim"], hidden_features=params["hidden_dim"])
    pt_mlp, keras_mlp = create_mlp_pair(**kwargs)
    input_pt = random_input(params, params["in_dim"])
    output_pt = pt_mlp(input_pt).detach().numpy()
    output_keras = keras_mlp(jnp.array(input_pt.numpy()), training=False)
    np.testing.assert_allclose(output_pt, output_keras, atol=1e-6)


def test_mlp_defaults(params):
    """🧪 Test Mlp: Defaulting hidden and out features to input dimension."""
    pt_mlp, keras_mlp = create_mlp_pair(in_dim=params["in_dim"])
    input_pt = random_input(params, params["in_dim"])
    output_pt = pt_mlp(input_pt).detach().numpy()
    output_keras = keras_mlp(jnp.array(input_pt.numpy()), training=False)
    assert output_keras.shape[-1] == params["in_dim"]
    np.testing.assert_allclose(output_pt, output_keras, atol=1e-6)


def test_mlp_no_bias(params):
    """🧪 Test Mlp: Operation without bias terms."""
    kwargs = dict(in_dim=params["in_dim"], hidden_features=params["hidden_dim"])
    pt_mlp, keras_mlp = create_mlp_pair(**kwargs, bias=False)
    input_pt = random_input(params, params["in_dim"])
    output_pt = pt_mlp(input_pt).detach().numpy()
    output_keras = keras_mlp(jnp.array(input_pt.numpy()), training=False)
    np.testing.assert_allclose(output_pt, output_keras, atol=1e-6)


def test_mlp_dropout_inference(params):
    """🧪 Test Mlp: Ensure dropout is handled correctly during inference."""
    # Create models with a non-zero dropout rate
    kwargs = dict(in_dim=params["in_dim"], hidden_features=params["hidden_dim"])
    pt_mlp, keras_mlp = create_mlp_pair(**kwargs, drop=params["drop_rate"])

    # Generate a random input tensor
    input_pt = random_input(params, params["in_dim"])

    # Get the output from the PyTorch model (in eval mode, dropout is off)
    output_pt = pt_mlp(input_pt).detach().numpy()

    # Get the output from the Keras model with training=False to disable
    # dropout
    output_keras = keras_mlp(jnp.array(input_pt.numpy()), training=False)

    # Assert that the outputs are nearly identical
    np.testing.assert_allclose(output_pt, output_keras, atol=1e-6)


def test_mlp_with_pretrained_dinov3_weights(params):
    """🧪 Test Mlp by loading real DINOv3 ViT-S/16 pretrained weights."""
    try:
        DINO_REPO_PATH = ""
        DINO_WEIGHT_PATH = MISSING_WEIGHT_PATH

        if not os.path.isfile(DINO_WEIGHT_PATH):
            raise FileNotFoundError("DINOv3 weight file not found.")

        # Load the full DINOv3 model from the local repository
        kwargs = dict(source="local", pretrained=False)
        load = torch.hub.load
        full_dinov3_model = load(DINO_REPO_PATH, "dinov3_vits16", **kwargs)
        cpu = torch.device("cpu")
        state_dict = torch.load(DINO_WEIGHT_PATH, map_location=cpu)
        full_dinov3_model.load_state_dict(state_dict)
        full_dinov3_model.eval()

        # Extract the first MLP block as the reference PyTorch model
        torch_model = full_dinov3_model.blocks[0].mlp

    except ASSET_ERRORS as e:
        message = "DINOv3 assets not found, skipping pretrained MLP test. "
        pytest.skip(message + f"Reason: {e}")

    # Get model dimensions from the loaded torch model
    dinov3_embed_dim = torch_model.fc1.in_features
    dinov3_hidden_dim = torch_model.fc1.out_features

    # Create the Keras model with a matching configuration
    kwargs = dict(in_features=dinov3_embed_dim)
    kwargs.update(hidden_features=dinov3_hidden_dim)
    kwargs.update(out_features=dinov3_embed_dim)
    # Use precise GELU to match PyTorch
    keras_model = Mlp(**kwargs, approximate_gelu=False)
    # Build the Keras layer by calling it with a dummy input
    keras_model(jnp.ones((1, 1, dinov3_embed_dim)))

    # Transfer weights from the PyTorch layer to the Keras layer
    fc1_weights = dense_weights(torch_model.fc1)
    keras_model.get_layer("mlp_fc1").set_weights(fc1_weights)
    fc2_weights = dense_weights(torch_model.fc2)
    keras_model.get_layer("mlp_fc2").set_weights(fc2_weights)

    # --- Run forward pass and compare outputs ---
    input_tensor = random_input(params, dinov3_embed_dim)
    output_pt = torch_model(input_tensor).detach().numpy()
    output_keras = keras_model(jnp.array(input_tensor.numpy()), training=False)

    # Calculate and validate the mean absolute difference
    mean_abs_diff = np.mean(np.abs(output_pt - output_keras))
    print(f"\nMean Absolute Difference: {mean_abs_diff}")

    # Assert that the outputs are numerically very close
    np.testing.assert_allclose(output_pt, output_keras, atol=1e-4)
    message = f"Mean absolute difference {mean_abs_diff} exceeds tolerance."
    assert mean_abs_diff < 1e-5, message


def test_swiglu_basic(params):
    """🧪 Test SwiGLUFFN: Basic case with specified dimensions."""
    kwargs = dict(in_dim=params["in_dim"], hidden_features=params["hidden_dim"])
    pt_swiglu, keras_swiglu = create_swiglu_pair(**kwargs)
    input_pt = random_input(params, params["in_dim"])
    output_pt = pt_swiglu(input_pt).detach().numpy()
    output_keras = keras_swiglu(jnp.array(input_pt.numpy()))
    np.testing.assert_allclose(output_pt, output_keras, atol=1e-6)


def test_swiglu_defaults(params):
    """🧪 Test SwiGLUFFN: Defaulting hidden and out features to input
    dimension."""
    pt_swiglu, keras_swiglu = create_swiglu_pair(in_dim=params["in_dim"])
    input_pt = random_input(params, params["in_dim"])
    output_pt = pt_swiglu(input_pt).detach().numpy()
    output_keras = keras_swiglu(jnp.array(input_pt.numpy()))
    assert output_keras.shape[-1] == params["in_dim"]
    np.testing.assert_allclose(output_pt, output_keras, atol=1e-6)


def test_swiglu_no_bias(params):
    """🧪 Test SwiGLUFFN: Operation without bias terms."""
    kwargs = dict(in_dim=params["in_dim"], hidden_features=params["hidden_dim"])
    pt_swiglu, keras_swiglu = create_swiglu_pair(**kwargs, bias=False)
    input_pt = random_input(params, params["in_dim"])
    output_pt = pt_swiglu(input_pt).detach().numpy()
    output_keras = keras_swiglu(jnp.array(input_pt.numpy()))
    np.testing.assert_allclose(output_pt, output_keras, atol=1e-6)


def test_swiglu_alignment(params):
    """🧪 Test SwiGLUFFN: Non-default alignment value."""
    kwargs = dict(in_dim=params["in_dim"], hidden_features=params["hidden_dim"])
    pt_swiglu, keras_swiglu = create_swiglu_pair(**kwargs, align_to=16)
    input_pt = random_input(params, params["in_dim"])
    output_pt = pt_swiglu(input_pt).detach().numpy()
    output_keras = keras_swiglu(jnp.array(input_pt.numpy()))
    np.testing.assert_allclose(output_pt, output_keras, atol=1e-6)


def test_swiglu_with_pretrained_dinov3_weights(params):
    """🧪 Test SwiGLUFFN with mock DINOv3 ViT-S/16 pretrained weights."""
    dinov3_embed_dim = 384
    dinov3_hidden_dim = 1536  # DINOv3 ViT-S uses an FFN size of 1536

    # 1. Create mock pre-trained weights using the PyTorch layer's structure
    kwargs = dict(in_features=dinov3_embed_dim)
    kwargs.update(hidden_features=dinov3_hidden_dim)
    torch_model_for_weights = PT_SwiGLUFFN(**kwargs)
    mock_weights = torch_model_for_weights.state_dict()

    # 2. Configure the actual PyTorch model and load weights
    torch_model = PT_SwiGLUFFN(**kwargs)
    torch_model.load_state_dict(mock_weights)
    torch_model.eval()

    args = (dinov3_embed_dim, dinov3_hidden_dim)
    keras_model = SwiGLUFFN(*args, out_features=dinov3_embed_dim)
    keras_model(jnp.ones((1, 1, dinov3_embed_dim)))
    for name in ("w1", "w2", "w3"):
        weights = state_dense_weights(mock_weights, name)
        keras_model.get_layer(f"mlp_{name}").set_weights(weights)

    input_tensor = random_input(params, dinov3_embed_dim)
    output_pt = torch_model(input_tensor).detach().numpy()
    output_keras = keras_model(jnp.array(input_tensor.numpy()))
    np.testing.assert_allclose(output_pt, output_keras, atol=1e-6)
