from typing import Callable, Dict

import numpy as np
import pytest
import tensorflow as tf
import torch

from neural_networks.losses import RMSELoss
from neural_networks.nn import Dense
from neural_networks.optimizers import SGD
from utils import check_closeness

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
    "activation_str", ["identity", "relu", "sigmoid", "tanh"])
def test_activations(activation_str: str) -> None:
    epochs = 10
    batch_size = 3
    np.random.seed(65)

    x = np.random.randint(low=0, high=10, size=(batch_size, 5))
    y = np.random.randint(low=0, high=10, size=(batch_size, 1))
    print(x, y)
    print("inshape", y.shape)
    hidden_layers_size = [3, 12, 7]
    layers = [x.shape[1]] + hidden_layers_size + [1]
    n_layers = len(layers) - 1
    dense_layers = []
    tf_weights_list = []
    tf_biases_list = []
    torch_weights_list = []
    torch_biases_list = []

    for idx in range(n_layers):
        dense = Dense(
            in_features=layers[idx],
            out_features=layers[idx+1],
            activation=activation_str)
        w = dense._weights.copy()
        b = dense._bias.copy()
        dense_layers.append(dense)

        w_tf = tf.Variable(w.astype(np.float32))
        b_tf = tf.Variable(b.astype(np.float32))
        tf_weights_list.append(w_tf)
        tf_biases_list.append(b_tf)

        w_torch = torch.tensor(w.astype(np.float32), requires_grad=True)
        b_torch = torch.tensor(b.astype(np.float32), requires_grad=True)
        torch_weights_list.append(w_torch)
        torch_biases_list.append(b_torch)

    x_tf = tf.constant(x.astype(np.float32))
    y_tf = tf.constant(y.astype(np.float32))
    x_torch = torch.tensor(x.astype(np.float32))
    y_torch = torch.tensor(y.astype(np.float32))

    loss = RMSELoss()
    activation_tf = TF_ACTIVATIONS_MAP[activation_str]
    activation_torch = TORCH_ACTIVATIONS_MAP[activation_str]

    optimizer = SGD(learning_rate=0.001)
    optimizer_tf = tf.keras.optimizers.SGD(learning_rate=0.001)
    optimizer_torch = torch.optim.SGD(
        params=[*torch_weights_list, *torch_biases_list], lr=0.001)
    for epoch in range(epochs):
        # Our neural network
        feed_in = x
        for idx in range(n_layers):
            output = dense_layers[idx].forward(inputs=feed_in)
            feed_in = output
        print("output", output.shape)
        cost_nn = loss.forward(y_pred=output, y_true=y)
        dL = loss.backprop()
        print("dl.shape", dL.shape)
        derivative = dL
        optimizer.set_cur_epoch(epoch + 1)
        for idx in range(n_layers - 1, -1, -1):
            derivative = dense_layers[idx].backprop(
                derivative, optimizer=optimizer)

        # Tensorflow neural network
        feed_in = x_tf
        with tf.GradientTape() as tape:
            for idx in range(n_layers):
                output = tf.matmul(
                    feed_in, tf_weights_list[idx]) + tf_biases_list[idx]
                output = activation_tf(output)
                feed_in = output
            cost_tf = tf.sqrt(tf.losses.mean_squared_error(output, y_tf))
        trainable_variables = [*tf_weights_list, *tf_biases_list]
        grads = tape.gradient(cost_tf, trainable_variables)
        optimizer_tf.apply_gradients(zip(grads, trainable_variables))

        # Pytorch neural network
        feed_in_torch = x_torch
        for idx in range(n_layers):
            optimizer_torch.zero_grad()
            output_torch = torch.matmul(
                feed_in_torch,
                torch_weights_list[idx]) + torch_biases_list[idx]
            output_torch = activation_torch(output_torch)
            feed_in_torch = output_torch
        loss_torch = torch.nn.MSELoss()
        loss_torch_fn = torch.sqrt(loss_torch(output_torch, y_torch))
        loss_torch_fn.backward()
        optimizer_torch.step()

        for idx in range(n_layers):
            assert check_closeness(
                dense_layers[idx]._weights, np.array(tf_weights_list[idx]))
            assert check_closeness(
                dense_layers[idx]._weights,
                torch_weights_list[idx].detach().numpy())
            assert check_closeness(
                dense_layers[idx]._bias, np.array(tf_biases_list[idx]))
            assert check_closeness(
                dense_layers[idx]._bias,
                torch_biases_list[idx].detach().numpy())
        assert np.allclose(cost_nn, cost_tf)
        assert np.allclose(cost_nn, loss_torch_fn.item())


@pytest.mark.parametrize("num_class", [2, 4, 10])
def test_softmax(num_class: int) -> None:
    epochs = 10
    batch_size = 1
    np.random.seed(65)

    x = np.random.randint(low=0, high=10, size=(batch_size, 5))

    y = np.zeros((batch_size, num_class))
    print(x, y)
    y[np.random.randint(low=0, high=num_class), 0] = 1

    hidden_layers_size = []
    layers = [x.shape[1]] + hidden_layers_size + [num_class]
    n_layers = len(layers) - 1
    dense_layers = []
    tf_weights_list = []
    tf_biases_list = []
    torch_weights_list = []
    torch_biases_list = []

    for idx in range(n_layers):
        dense = Dense(
            in_features=layers[idx],
            out_features=layers[idx+1],
            activation="softmax")
        w = dense._weights.copy()
        b = dense._bias.copy()
        dense_layers.append(dense)

        w_tf = tf.Variable(w.astype(np.float32))
        b_tf = tf.Variable(b.astype(np.float32))
        tf_weights_list.append(w_tf)
        tf_biases_list.append(b_tf)

        w_torch = torch.tensor(w.astype(np.float32), requires_grad=True)
        b_torch = torch.tensor(b.astype(np.float32), requires_grad=True)
        torch_weights_list.append(w_torch)
        torch_biases_list.append(b_torch)

    x_tf = tf.constant(x.astype(np.float32))
    y_tf = tf.constant(y.astype(np.float32))
    x_torch = torch.tensor(x.astype(np.float32))
    y_torch = torch.tensor(y.astype(np.float32))

    loss = RMSELoss()
    activation_tf = TF_ACTIVATIONS_MAP["softmax"]
    activation_torch = TORCH_ACTIVATIONS_MAP["softmax"]

    optimizer = SGD(learning_rate=0.001)
    optimizer_tf = tf.keras.optimizers.SGD(learning_rate=0.001)
    optimizer_torch = torch.optim.SGD(
        params=[*torch_weights_list, *torch_biases_list], lr=0.001)
    for epoch in range(epochs):
        # Our neural network
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
                derivative, optimizer=optimizer)

        # Tensorflow neural network
        feed_in = x_tf
        with tf.GradientTape() as tape:
            for idx in range(n_layers):
                output = tf.matmul(
                    feed_in, tf_weights_list[idx]) + tf_biases_list[idx]
                output = activation_tf(output)
                feed_in = output
            cost_tf = tf.sqrt(tf.losses.mean_squared_error(output, y_tf))
        trainable_variables = [*tf_weights_list, *tf_biases_list]
        grads = tape.gradient(cost_tf, trainable_variables)
        optimizer_tf.apply_gradients(zip(grads, trainable_variables))

        # Pytorch neural network
        feed_in_torch = x_torch
        for idx in range(n_layers):
            optimizer_torch.zero_grad()
            output_torch = torch.matmul(
                feed_in_torch,
                torch_weights_list[idx]) + torch_biases_list[idx]
            output_torch = activation_torch(output_torch)
            feed_in_torch = output_torch
        loss_torch = torch.nn.MSELoss()
        loss_torch_fn = torch.sqrt(loss_torch(output_torch, y_torch))
        loss_torch_fn.backward()
        optimizer_torch.step()

        for idx in range(n_layers):
            assert check_closeness(
                dense_layers[idx]._weights, np.array(tf_weights_list[idx]))
            assert check_closeness(
                dense_layers[idx]._weights,
                torch_weights_list[idx].detach().numpy())
            assert check_closeness(
                dense_layers[idx]._bias, np.array(tf_biases_list[idx]))
            assert check_closeness(
                dense_layers[idx]._bias,
                torch_biases_list[idx].detach().numpy())
        assert np.allclose(cost_nn, cost_tf)
        assert np.allclose(cost_nn, loss_torch_fn.item())
