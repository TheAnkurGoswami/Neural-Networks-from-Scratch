from typing import Callable, Dict, List

import numpy as np
import pytest
import tensorflow as tf
import torch

from neural_networks.losses import RMSELoss
from neural_networks.layers import Dense # Updated import
from neural_networks.optimizers import SGD
from tests.templates import (
    get_bias_template,
    get_loss_template,
    get_weight_template,
)
from utils import check_closeness

# Mapping of activation functions for TensorFlow and PyTorch
TF_ACTIVATIONS_MAP: Dict[str, Callable[[tf.Tensor], tf.Tensor]] = {
    "identity": lambda x: x,
    "relu": tf.nn.relu,
    "sigmoid": tf.nn.sigmoid,
    "tanh": tf.nn.tanh,
    "softmax": tf.nn.softmax,
}

TORCH_ACTIVATIONS_MAP: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "identity": lambda x: x,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "softmax": torch.softmax,
}


@pytest.mark.parametrize(
    "activation_str", ["identity", "relu", "sigmoid", "tanh"]
)
def test_activations(activation_str: str) -> None:
    """
    Test the activation functions for custom, TensorFlow, and PyTorch models.

    Args:
        activation_str (str): The activation function to test.
    """
    epochs: int = 10
    np.random.seed(65)
    torch.manual_seed(65)

    # Generate random input and output data
    x: np.ndarray = np.random.randint(low=0, high=10, size=(1, 5))
    y: np.ndarray = np.random.randint(low=0, high=10, size=(1, 1))
    hidden_layers_size: List[int] = [3, 12, 7]
    layers: List[int] = [x.shape[1]] + hidden_layers_size + [1]
    n_layers: int = len(layers) - 1
    dense_layers: List[Dense] = []
    tf_weights_list: List[tf.Variable] = []
    tf_biases_list: List[tf.Variable] = []
    torch_weights_list: List[torch.Tensor] = []
    torch_biases_list: List[torch.Tensor] = []

    # Initialize Dense layers and weights for custom, TensorFlow, and PyTorch
    # models
    for idx in range(n_layers):
        dense: Dense = Dense(
            in_features=layers[idx],
            out_features=layers[idx + 1],
            activation=activation_str,
        )
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
    x_tf: tf.Tensor = tf.constant(x.astype(np.float32))
    y_tf: tf.Tensor = tf.constant(y.astype(np.float32))
    x_torch: torch.Tensor = torch.tensor(x.astype(np.float32))
    y_torch: torch.Tensor = torch.tensor(y.astype(np.float32))

    # Define loss function and activation functions for TensorFlow and PyTorch
    loss: RMSELoss = RMSELoss()
    activation_tf: Callable[[tf.Tensor], tf.Tensor] = TF_ACTIVATIONS_MAP[
        activation_str
    ]
    activation_torch: Callable[[torch.Tensor], torch.Tensor] = (
        TORCH_ACTIVATIONS_MAP[activation_str]
    )

    # Define optimizers for custom, TensorFlow, and PyTorch models
    optimizer: SGD = SGD(learning_rate=0.001)
    optimizer_tf: tf.keras.optimizers.SGD = tf.keras.optimizers.SGD(
        learning_rate=0.001
    )
    optimizer_torch: torch.optim.SGD = torch.optim.SGD(
        params=[*torch_weights_list, *torch_biases_list], lr=0.001
    )

    for epoch in range(epochs):
        # Forward pass for custom neural network
        feed_in: np.ndarray = x
        for idx in range(n_layers):
            output: np.ndarray = dense_layers[idx].forward(inputs=feed_in)
            feed_in = output
        cost_nn: float = loss.forward(y_pred=output, y_true=y)
        dL: np.ndarray = loss.backprop()
        derivative: np.ndarray = dL
        optimizer.set_cur_epoch(epoch + 1)
        for idx in range(n_layers - 1, -1, -1):
            derivative = dense_layers[idx].backprop(
                derivative, optimizer=optimizer
            )

        # Forward pass for TensorFlow neural network
        feed_in_tf: tf.Tensor = x_tf
        with tf.GradientTape() as tape:
            for idx in range(n_layers):
                output_tf: tf.Tensor = (
                    tf.matmul(feed_in_tf, tf_weights_list[idx])
                    + tf_biases_list[idx]
                )
                output_tf = activation_tf(output_tf)
                feed_in_tf = output_tf
            loss_tf: tf.keras.losses.MeanSquaredError = (
                tf.keras.losses.MeanSquaredError()
            )
            cost_tf: tf.Tensor = tf.sqrt(loss_tf(output_tf, y_tf) + 1e-16)
        trainable_variables: List[tf.Variable] = [
            *tf_weights_list,
            *tf_biases_list,
        ]
        grads: List[tf.Tensor] = tape.gradient(cost_tf, trainable_variables)
        optimizer_tf.apply_gradients(
            zip(grads, trainable_variables, strict=False)
        )

        # Forward pass for PyTorch neural network
        feed_in_torch: torch.Tensor = x_torch
        for idx in range(n_layers):
            output_torch: torch.Tensor = (
                torch.matmul(feed_in_torch, torch_weights_list[idx])
                + torch_biases_list[idx]
            )
            output_torch = activation_torch(output_torch)
            feed_in_torch = output_torch
        loss_torch: torch.nn.MSELoss = torch.nn.MSELoss()
        loss_torch_fn: torch.Tensor = torch.sqrt(
            loss_torch(output_torch, y_torch) + 1e-16
        )
        loss_torch_fn.backward()
        optimizer_torch.step()
        optimizer_torch.zero_grad()

        # Check if weights and biases are close between custom, TensorFlow, and
        # PyTorch models
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
