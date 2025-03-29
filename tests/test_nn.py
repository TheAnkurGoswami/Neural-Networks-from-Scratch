from typing import List

import numpy as np
import pytest
import tensorflow as tf
import torch

from neural_networks.losses import CrossEntropyLoss, MSELoss, RMSELoss
from neural_networks.nn import Dense
from neural_networks.optimizers import get_optimizer
from tests.templates import (
    get_bias_template,
    get_loss_template,
    get_weight_template,
)
from utils import check_closeness
import logging
import os

torch.set_printoptions(precision=10)
np.set_printoptions(precision=10)
tf.keras.backend.set_floatx("float32")


def test_no_hidden_layer_simple_nn() -> None:
    # Set the number of epochs and learning rate for training
    epochs = 10
    learning_rate = 0.01
    # batch_size = 3
    np.random.seed(100)

    # Generate random input and output data
    x = np.random.randint(low=0, high=10, size=(1, 5)).astype(np.float32)
    y = np.random.randint(low=0, high=10, size=(1, 1)).astype(np.float32)

    # Initialize a single dense layer neural network
    dense = Dense(in_features=x.shape[1], out_features=1)
    w = dense._weights.clone().detach()
    b = dense._bias.clone().detach()

    # Convert data to TensorFlow tensors
    x_tf = tf.constant(x.astype(np.float32))
    y_tf = tf.constant(y.astype(np.float32))
    w_tf = tf.Variable(w)
    b_tf = tf.Variable(b)

    # Convert data to PyTorch tensors
    x_torch = torch.tensor(x.astype(np.float32))
    y_torch = torch.tensor(y.astype(np.float32))
    w_torch = w.clone().detach().requires_grad_(True)
    b_torch = b.clone().detach().requires_grad_(True)

    # Initialize optimizers and loss functions for each framework
    optimizer = get_optimizer("sgd")(learning_rate=learning_rate, momentum=0)
    loss = MSELoss()
    optimizer_torch = torch.optim.SGD(
        [w_torch, b_torch], lr=learning_rate, momentum=0
    )
    loss_torch = torch.nn.MSELoss()

    for _ in range(epochs):
        # Train the custom neural network
        y_pred = dense.forward(inputs=x)
        cost_nn = loss.forward(y_pred=y_pred, y_true=y)
        dL = loss.backprop()
        dense.backprop(dL, optimizer=optimizer)

        # Train the TensorFlow neural network
        optimizer_tf = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        loss_fn = tf.keras.losses.MeanSquaredError()
        with tf.GradientTape() as tape:
            y_hat = tf.matmul(x_tf, w_tf) + b_tf
            cost_tf = loss_fn(y_hat, y_tf)
        trainable_variables = [w_tf, b_tf]
        grads = tape.gradient(cost_tf, trainable_variables)
        optimizer_tf.apply_gradients(zip(grads, trainable_variables))

        # Train the PyTorch neural network
        optimizer_torch.zero_grad()
        y_pred = torch.matmul(x_torch, w_torch) + b_torch
        loss_torch_fn = loss_torch(y_pred, y_torch)
        loss_torch_fn.backward()
        optimizer_torch.step()

        # Check if the weights, biases, and costs are close across frameworks
        assert check_closeness(dense._weights.detach().numpy(), w_tf)
        assert check_closeness(
            dense._weights.detach().numpy(), w_torch.detach().numpy()
        )
        assert check_closeness(dense._bias.detach().numpy(), b_tf)
        assert check_closeness(
            dense._bias.detach().numpy(), b_torch.detach().numpy()
        )
        assert check_closeness(cost_nn.detach().numpy(), cost_tf)
        assert check_closeness(loss_torch_fn.item(), cost_nn.detach().numpy())


# FIXME: add different regression loss tests
@pytest.mark.parametrize("hidden_layers_size", [[5], [2, 3], [6, 4, 10]])
@pytest.mark.parametrize("batch_size", [1, 10, 64])
@pytest.mark.parametrize("normalize_inputs", [True, False])
@pytest.mark.parametrize("output_neurons", [1])
# FIXME: try with multiple output neurons
def test_n_hidden_layer_simple_nn(
    hidden_layers_size: List[int],
    batch_size: int,
    normalize_inputs: bool,
    output_neurons: int,
) -> None:
    # Set the number of epochs and learning rate for training
    epochs = 10
    learning_rate = 0.001
    np.random.seed(100)

    # Generate random input and output data
    if not normalize_inputs:
        x = np.random.randint(low=0, high=10, size=(batch_size, 5))
    else:
        x = np.random.randint(low=0, high=10, size=(batch_size, 5)).astype(
            np.float32
        )
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(
            x, axis=1, keepdims=True
        )
    y = np.random.randint(low=0, high=10, size=(batch_size, output_neurons))

    # Define the architecture of the neural network
    layers = [x.shape[1]] + hidden_layers_size + [output_neurons]
    n_layers = len(layers) - 1
    dense_layers = []
    tf_weights_list = []
    tf_biases_list = []
    torch_weights_list = []
    torch_biases_list = []

    # Initialize layers and weights for each framework
    for idx in range(n_layers):
        if idx == n_layers - 1:
            dense = Dense(
                in_features=layers[idx],
                out_features=layers[idx + 1],
                # activation="relu",
            )
        else:
            dense = Dense(
                in_features=layers[idx],
                out_features=layers[idx + 1],
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

    # Convert data to TensorFlow tensors
    x_tf = tf.constant(x.astype(np.float32))
    y_tf = tf.constant(y.astype(np.float32))

    # Convert data to PyTorch tensors
    x_torch = torch.tensor(x.astype(np.float32))
    y_torch = torch.tensor(y.astype(np.float32))

    # Initialize loss functions and optimizers for each framework
    loss = RMSELoss()
    optimizer = get_optimizer("sgd")(learning_rate=learning_rate, momentum=0)
    optimizer_tf = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    optimizer_torch = torch.optim.SGD(
        params=[*torch_weights_list, *torch_biases_list], lr=learning_rate
    )

    for epoch in range(epochs):
        # Train the custom neural network
        feed_in = x
        for idx in range(n_layers):
            output = dense_layers[idx].forward(inputs=feed_in)
            feed_in = output
        cost_nn = loss.forward(y_pred=output, y_true=y)
        dL = loss.backprop()
        derivative = dL
        for idx in range(n_layers - 1, -1, -1):
            derivative = dense_layers[idx].backprop(
                derivative, optimizer=optimizer
            )

        # Train the TensorFlow neural network
        feed_in = x_tf
        with tf.GradientTape() as tape:
            for idx in range(n_layers):
                output = (
                    tf.matmul(feed_in, tf_weights_list[idx])
                    + tf_biases_list[idx]
                )
                # if idx != n_layers - 1:
                #     output = tf.nn.relu(output)
                feed_in = output
            loss_tf = tf.keras.losses.MeanSquaredError()
            cost_tf = tf.sqrt(loss_tf(output, y_tf))
        trainable_variables = [*tf_weights_list, *tf_biases_list]
        grads = tape.gradient(cost_tf, trainable_variables)
        optimizer_tf.apply_gradients(zip(grads, trainable_variables))

        # Train the PyTorch neural network
        feed_in = x_torch
        for idx in range(n_layers):
            optimizer_torch.zero_grad()
            output = (
                torch.matmul(feed_in, torch_weights_list[idx])
                + torch_biases_list[idx]
            )
            # if idx != n_layers - 1:
            #     output = torch.relu(output)
            feed_in = output
        loss_torch = torch.nn.MSELoss()
        loss_torch_fn = torch.sqrt(loss_torch(output, y_torch))
        loss_torch_fn.backward()
        optimizer_torch.step()

        # Check if the weights, biases, and costs are close across frameworks
        for idx in range(n_layers):
            # print(dense_layers[idx]._weights, np.array(tf_weights_list[idx]))
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

        # print("loss", epoch, cost_nn, cost_tf, loss_torch_fn.item())


@pytest.mark.parametrize("hidden_layers_size", [[5], [2, 3], [6, 4, 10]])
@pytest.mark.parametrize("batch_size", [1, 2, 64])
@pytest.mark.parametrize("n_classes", [3, 5])
@pytest.mark.parametrize("normalize_inputs", [True, False])
def test_n_hidden_layer_classification(
    hidden_layers_size: List[int],
    batch_size: int,
    n_classes: int,
    normalize_inputs: bool,
) -> None:

    # Set the number of epochs and learning rate for training
    epochs = 10
    learning_rate = 0.001
    # batch_size = 3
    np.random.seed(100)
    torch.manual_seed(100)

    # Generate random input and output data
    if not normalize_inputs:
        x = np.random.randint(low=0, high=10, size=(batch_size, 5))
    else:
        x = np.random.randint(low=0, high=10, size=(batch_size, 5)).astype(
            np.float32
        )
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(
            x, axis=1, keepdims=True
        )
    y = np.random.randint(low=0, high=n_classes, size=(batch_size, 1))
    yhot = np.zeros((y.shape[0], n_classes), dtype=np.int8)
    for ix, cls in enumerate(y):
        yhot[ix, cls] = 1
    print(yhot)
    # Define the architecture of the neural network
    layers = [x.shape[1]] + hidden_layers_size + [n_classes]
    n_layers = len(layers) - 1
    dense_layers = []
    tf_weights_list = []
    tf_biases_list = []
    torch_weights_list = []
    torch_biases_list = []

    # Initialize layers and weights for each framework
    for idx in range(n_layers):
        if idx == n_layers - 1:
            dense = Dense(
                in_features=layers[idx],
                out_features=layers[idx + 1],
                activation="softmax",
                retain_grad=True,
            )
        else:
            dense = Dense(
                in_features=layers[idx],
                out_features=layers[idx + 1],
                activation="relu",
                retain_grad=True,
            )
        # w = dense._weights.copy()
        # b = dense._bias.copy()
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
    #     print(w_torch, w_torch.detach().numpy())
    # assert False
    # Convert data to TensorFlow tensors
    x_tf = tf.constant(x.astype(np.float32))
    y_tf = tf.constant(yhot.astype(np.float32))

    # Convert data to PyTorch tensors
    x_torch = torch.tensor(x.astype(np.float32))
    y_torch = torch.tensor(yhot.astype(np.float32))

    # Initialize loss functions and optimizers for each framework
    loss = CrossEntropyLoss()
    optimizer = get_optimizer("sgd")(learning_rate=learning_rate, momentum=0)
    optimizer_tf = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    optimizer_torch = torch.optim.SGD(
        params=[*torch_weights_list, *torch_biases_list], lr=learning_rate
    )

    for epoch in range(epochs):
        # Train the custom neural network
        feed_in = x
        for idx in range(n_layers):
            output = dense_layers[idx].forward(inputs=feed_in)
            feed_in = output
        # print("output", output, yhot)
        cost_nn = loss.forward(y_pred=output, y_true=yhot)
        dL = loss.backprop()
        derivative = dL
        print("dL", derivative)
        for idx in range(n_layers - 1, -1, -1):
            derivative = dense_layers[idx].backprop(
                derivative, optimizer=optimizer
            )

        # Train the TensorFlow neural network
        feed_in = x_tf
        with tf.GradientTape() as tape:
            for idx in range(n_layers):
                output = (
                    tf.matmul(feed_in, tf_weights_list[idx])
                    + tf_biases_list[idx]
                )
                if idx == n_layers - 1:
                    output = tf.nn.softmax(output)
                else:
                    output = tf.nn.relu(output)
                feed_in = output
            loss_tf = tf.keras.losses.CategoricalCrossentropy()
            # print("output", output, y_tf)
            cost_tf = loss_tf(y_tf, output)
        trainable_variables = [*tf_weights_list, *tf_biases_list]
        grads = tape.gradient(cost_tf, trainable_variables)
        # print("grads", grads)
        optimizer_tf.apply_gradients(zip(grads, trainable_variables))

        # Train the PyTorch neural network
        feed_in = x_torch
        torch_outputs = []
        activations = []
        for idx in range(n_layers):
            optimizer_torch.zero_grad()
            output = (
                torch.matmul(feed_in, torch_weights_list[idx])
                + torch_biases_list[idx]
            )
            torch_outputs.append(output)
            torch_outputs[
                -1
            ].retain_grad()  # Retain gradients for non-leaf tensors
            if idx == n_layers - 1:
                output = torch.softmax(output, dim=1)
                # print("output", output, y_torch)
                output = torch.clip(
                    output, 1e-07, 1.0 - 1e-07
                )  # numerical stability
            else:
                output = torch.relu(output)
            activations.append(output)
            activations[
                -1
            ].retain_grad()  # Retain gradients for non-leaf tensors
            feed_in = output
        loss_torch = torch.nn.CrossEntropyLoss()
        # Log to nullify the effect of having Softmax inside CorssEntropyLoss
        log_probs = torch.log(output)
        loss_torch_fn = loss_torch(log_probs, y_torch)
        loss_torch_fn.retain_grad()  # Retain gradients for non-leaf tensors
        loss_torch_fn.backward()
        print("dL", loss_torch_fn.grad)
        optimizer_torch.step()

        for idx, (cus_layer, pt_layer) in enumerate(
            zip(dense_layers, torch_outputs)
        ):
            logging.info(
                f"Epoch {epoch}, Layer {idx + 1} - dA: {activations[idx].shape}, Gradient: {activations[idx].grad}"
            )
            logging.info(
                f"Epoch {epoch}, Layer {idx + 1} - dZ (Custom): {cus_layer._dZ.shape}, Values: {cus_layer._dZ}"
            )
            logging.info(
                f"Epoch {epoch}, Layer {idx + 1} - dZ (PyTorch): {pt_layer.shape}, Gradient: {pt_layer.grad}"
            )
            logging.info(
                f"Epoch {epoch}, Layer {idx + 1} - dW (Custom): {cus_layer._dW.shape}, Values: {cus_layer._dW}"
            )
            logging.info(
                f"Epoch {epoch}, Layer {idx + 1} - dW (PyTorch): {torch_weights_list[idx].shape}, Gradient: {torch_weights_list[idx].grad}"
            )
            logging.info(
                f"Epoch {epoch}, Layer {idx + 1} - dB (Custom): {cus_layer._dB.shape}, Values: {cus_layer._dB}"
            )
            logging.info(
                f"Epoch {epoch}, Layer {idx + 1} - dB (PyTorch): {torch_biases_list[idx].shape}, Gradient: {torch_biases_list[idx].grad}"
            )
        logging.info(
            f"Loss at epoch {epoch}: Custom NN: {cost_nn}, TensorFlow: {cost_tf}, PyTorch: {loss_torch_fn.item()}"
        )
        # assert False

        # Check if the weights, biases, and costs are close across frameworks
        for idx in range(n_layers):
            logging.info(
                f"Layer {idx + 1}: Custom Weights: {dense_layers[idx]._weights}, TensorFlow Weights: {tf_weights_list[idx]}"
            )
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

            logging.info(
                f"Layer {idx + 1}: Custom Bias: {dense_layers[idx]._bias}, TensorFlow Bias: {tf_biases_list[idx]}"
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
