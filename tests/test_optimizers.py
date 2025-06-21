from typing import Any, Dict, List

import numpy as np
import pytest
import tensorflow as tf
import torch

from neural_networks.losses import RMSELoss
from neural_networks.nn import Dense
from neural_networks.optimizers import get_optimizer
from tests.templates import (
    get_bias_template,
    get_loss_template,
    get_weight_template,
)
from utils import check_closeness

# Mapping of optimizer strings to TensorFlow optimizers
TF_OPTIM_MAP = {
    "sgd": tf.keras.optimizers.SGD,
    "rmsprop": tf.keras.optimizers.RMSprop,
    "adam": tf.keras.optimizers.Adam,
}

# Mapping of optimizer strings to PyTorch optimizers
TORCH_OPTIM_MAP = {
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "adam": torch.optim.Adam,
}

# Set print options for better readability
# torch.set_printoptions(precision=8)
# np.set_printoptions(precision=8)
# tf.keras.backend.set_floatx("float32")


@pytest.mark.parametrize(
    "optimizer_str, kwargs",
    [
        ("sgd", {"learning_rate": 0.001, "momentum": 0.9}),
        ("rmsprop", {"learning_rate": 0.001, "rho": 0.9, "epsilon": 1e-07}),
        (
            "adam",
            {
                "learning_rate": 0.001,
                "beta_1": 0.9,
                "beta_2": 0.999,
                "epsilon": 1e-07,
            },
        ),
    ],
)
def test_optimizer(optimizer_str: str, kwargs: Dict[str, Any]) -> None:
    """
    Test the optimizer implementation against TensorFlow and PyTorch
    optimizers.

    Args:
        optimizer_str (str): The optimizer name.
        kwargs (Dict[str, Any]): The optimizer parameters.
    """
    epochs = 10
    # np.random.seed(65)
    # torch.manual_seed(65)

    # Generate random input and output data
    x = np.random.randint(low=0, high=10, size=(1, 5))
    y = np.random.randint(low=0, high=10, size=(1, 1))

    hidden_layers_size = [3, 12, 7]
    layers = [x.shape[1]] + hidden_layers_size + [1]
    n_layers = len(layers) - 1
    dense_layers: List[Dense] = []
    tf_weights_list = []
    tf_biases_list = []
    torch_weights_list = []
    torch_biases_list = []

    # Initialize custom optimizer
    optimizer = get_optimizer(optimizer_str)(**kwargs)

    # Initialize layers and weights for custom, TensorFlow, and PyTorch
    # implementations
    for idx in range(n_layers):
        dense = Dense(in_features=layers[idx], out_features=layers[idx + 1])
        w = dense._weights.clone().detach()
        b = dense._bias.clone().detach()
        dense_layers.append(dense)

        w_tf = tf.Variable(w)
        b_tf = tf.Variable(b)
        tf_weights_list.append(w_tf)
        tf_biases_list.append(b_tf)

        w_torch = w.clone().detach().requires_grad_(True)
        b_torch = b.clone().detach().requires_grad_(True)
        torch_weights_list.append(w_torch)
        torch_biases_list.append(b_torch)

    # Convert input and output data to TensorFlow and PyTorch tensors
    x_tf = tf.constant(x.astype(np.float32))
    y_tf = tf.constant(y.astype(np.float32))
    x_torch = torch.tensor(x.astype(np.float32), requires_grad=True)
    y_torch = torch.tensor(y.astype(np.float32))

    # Initialize loss function and optimizers for TensorFlow and PyTorch
    loss = RMSELoss()
    optimizer_tf = TF_OPTIM_MAP[optimizer_str](**kwargs)
    new_kwargs = kwargs.copy()
    torch_tf_map = {"learning_rate": "lr", "epsilon": "eps", "rho": "alpha"}
    new_kwargs = {}
    for key in kwargs:
        try:
            new_kwargs[torch_tf_map[key]] = kwargs[key]
        except KeyError:
            if key not in {"beta_1", "beta_2"}:
                new_kwargs[key] = kwargs[key]
    if "beta_1" in kwargs.keys() and "beta_2" in kwargs.keys():
        new_kwargs["betas"] = (kwargs["beta_1"], kwargs["beta_2"])

    optimizer_torch = TORCH_OPTIM_MAP[optimizer_str](
        params=[*torch_weights_list, *torch_biases_list], **new_kwargs
    )

    # Training loop
    for epoch in range(epochs):
        # Custom neural network
        feed_in = x
        for idx in range(n_layers):
            output = dense_layers[idx].forward(inputs=feed_in)
            feed_in = output
        cost_nn = loss.forward(y_pred=output, y_true=y)
        dL = loss.backprop()
        derivative = dL
        optimizer.set_cur_epoch(epoch + 1)
        for idx in range(n_layers - 1, -1, -1):
            derivative = dense_layers[idx].backprop(
                derivative, optimizer=optimizer
            )

        # TensorFlow neural network
        feed_in = x_tf
        with tf.GradientTape() as tape:
            for idx in range(n_layers):
                output = (
                    tf.matmul(feed_in, tf_weights_list[idx])
                    + tf_biases_list[idx]
                )
                feed_in = output
            loss_fn = tf.keras.losses.MeanSquaredError()
            cost_tf = tf.sqrt(loss_fn(output, y_tf))
        trainable_variables = [*tf_weights_list, *tf_biases_list]
        grads = tape.gradient(cost_tf, trainable_variables)
        optimizer_tf.apply_gradients(
            zip(grads, trainable_variables, strict=False)
        )

        # PyTorch neural network
        feed_in_torch = x_torch
        for idx in range(n_layers):
            optimizer_torch.zero_grad()
            output_torch = (
                torch.matmul(feed_in_torch, torch_weights_list[idx])
                + torch_biases_list[idx]
            )
            feed_in_torch = output_torch
        loss_torch = torch.nn.MSELoss()
        loss_torch_fn = torch.sqrt(loss_torch(output_torch, y_torch))
        loss_torch_fn.backward()
        optimizer_torch.step()

        # Check closeness of weights, biases, and loss between implementations
        for idx in range(n_layers):
            assert check_closeness(
                dense_layers[idx]._weights.detach().numpy(),
                tf_weights_list[idx],
            ), (
                f"Epoch: {epoch}, Layer: {idx + 1} - "
                f"{get_weight_template('tf')}"
            )
            assert check_closeness(
                dense_layers[idx]._weights.detach().numpy(),
                torch_weights_list[idx].detach().numpy(),
            ), (
                f"Epoch: {epoch}, Layer: {idx + 1} - "
                f"{get_weight_template('pt')}"
            )

            assert check_closeness(
                dense_layers[idx]._bias.detach().numpy(), tf_biases_list[idx]
            ), f"Epoch: {epoch}, Layer: {idx + 1} - {get_bias_template('tf')}"
            assert check_closeness(
                dense_layers[idx]._bias.detach().numpy(),
                torch_biases_list[idx].detach().numpy(),
            ), f"Epoch: {epoch}, Layer: {idx + 1} - {get_bias_template('pt')}"
        assert check_closeness(
            cost_nn.detach().numpy(), cost_tf
        ), f"Epoch: {epoch}, Layer: {idx + 1} - {get_loss_template('tf')}"
        assert check_closeness(
            cost_nn.detach().numpy(), loss_torch_fn.item()
        ), f"Epoch: {epoch}, Layer: {idx + 1} - {get_loss_template('pt')}"
