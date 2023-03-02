from typing import List

import numpy as np
import pytest
import tensorflow as tf
import torch

from neural_networks.losses import MSELoss, RMSELoss
from neural_networks.nn import Dense
from neural_networks.optimizers import get_optimizer
from utils import check_closeness


def test_no_hidden_layer_simple_nn() -> None:
    epochs = 10
    learning_rate = 0.01
    batch_size = 3
    np.random.seed(100)

    x = np.random.randint(low=0, high=10, size=(batch_size, 5))
    y = np.random.randint(low=0, high=10, size=(batch_size, 1))

    dense = Dense(in_features=x.shape[1], out_features=1)
    w = dense._weights.copy()
    b = dense._bias.copy()

    x_tf = tf.constant(x.astype(np.float32))
    y_tf = tf.constant(y.astype(np.float32))
    w_tf = tf.Variable(w.astype(np.float32))
    b_tf = tf.Variable(b.astype(np.float32))

    x_torch = torch.tensor(x.astype(np.float32))
    y_torch = torch.tensor(y.astype(np.float32))
    w_torch = torch.tensor(w.astype(np.float32), requires_grad=True)
    b_torch = torch.tensor(b.astype(np.float32), requires_grad=True)

    optimizer = get_optimizer("sgd")(learning_rate=learning_rate,  momentum=0)
    loss = MSELoss()
    optimizer_torch = torch.optim.SGD(
        [w_torch, b_torch], lr=learning_rate, momentum=0)
    loss_torch = torch.nn.MSELoss()
    for _ in range(epochs):
        # Our neural network
        y_pred = dense.forward(inputs=x)
        cost_nn = loss.forward(y_pred=y_pred, y_true=y)
        dL = loss.backprop()
        dense.backprop(dL, optimizer=optimizer)

        # Tensorflow neural network
        optimizer_tf = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        loss_fn = tf.keras.losses.MeanSquaredError()
        with tf.GradientTape() as tape:
            y_hat = tf.matmul(x_tf, w_tf) + b_tf
            cost_tf = loss_fn(y_hat, y_tf)
        trainable_variables = [w_tf, b_tf]
        grads = tape.gradient(cost_tf, trainable_variables)
        optimizer_tf.apply_gradients(zip(grads, trainable_variables))

        # Pytorch neural network
        optimizer_torch.zero_grad()
        y_pred = torch.matmul(x_torch, w_torch) + b_torch
        loss_torch_fn = loss_torch(y_pred, y_torch)
        loss_torch_fn.backward()
        optimizer_torch.step()

        assert check_closeness(dense._weights, w_tf)
        assert check_closeness(
            dense._weights, w_torch.detach().numpy())
        print(dense._bias, b_tf)
        assert check_closeness(dense._bias, b_tf)
        assert check_closeness(dense._bias, b_torch.detach().numpy())
        assert check_closeness(cost_nn, cost_tf)
        assert check_closeness(loss_torch_fn.item(), cost_nn)


@pytest.mark.parametrize("hidden_layers_size", [[5], [2, 3], [6, 4, 10]])
def test_n_hidden_layer_simple_nn(hidden_layers_size: List[int]) -> None:
    epochs = 10
    learning_rate = 0.001
    batch_size = 3
    np.random.seed(100)

    x = np.random.randint(low=0, high=10, size=(batch_size, 5))
    y = np.random.randint(low=0, high=10, size=(batch_size, 1))

    layers = [x.shape[1]] + hidden_layers_size + [1]
    n_layers = len(layers) - 1
    dense_layers = []
    tf_weights_list = []
    tf_biases_list = []
    torch_weights_list = []
    torch_biases_list = []

    for idx in range(n_layers):
        dense = Dense(in_features=layers[idx], out_features=layers[idx+1])
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
    optimizer = get_optimizer("sgd")(learning_rate=learning_rate, momentum=0)
    optimizer_tf = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    optimizer_torch = torch.optim.SGD(
        params=[*torch_weights_list, *torch_biases_list], lr=learning_rate)
    for _ in range(epochs):
        # Our neural network
        feed_in = x
        for idx in range(n_layers):
            output = dense_layers[idx].forward(inputs=feed_in)
            feed_in = output
        cost_nn = loss.forward(y_pred=output, y_true=y)
        dL = loss.backprop()
        derivative = dL
        for idx in range(n_layers - 1, -1, -1):
            derivative = dense_layers[idx].backprop(
                derivative, optimizer=optimizer)

        # Tensorflow neural network
        feed_in = x_tf
        with tf.GradientTape() as tape:
            for idx in range(n_layers):
                output = tf.matmul(
                    feed_in, tf_weights_list[idx]) + tf_biases_list[idx]
                feed_in = output
            cost_tf = tf.sqrt(tf.losses.mean_squared_error(output, y_tf))
        trainable_variables = [*tf_weights_list, *tf_biases_list]
        grads = tape.gradient(cost_tf, trainable_variables)
        print(grads)
        optimizer_tf.apply_gradients(zip(grads, trainable_variables))

        # Pytorch neural network
        feed_in = x_torch
        for idx in range(n_layers):
            optimizer_torch.zero_grad()
            output = torch.matmul(
                feed_in, torch_weights_list[idx]) + torch_biases_list[idx]
            feed_in = output
        loss_torch = torch.nn.MSELoss()
        loss_torch_fn = torch.sqrt(loss_torch(output, y_torch))
        loss_torch_fn.backward()
        optimizer_torch.step()

        for idx in range(n_layers):
            print(dense_layers[idx]._weights, np.array(tf_weights_list[idx]))
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
        assert check_closeness(cost_nn, cost_tf)
        assert check_closeness(cost_nn, loss_torch_fn.detach().numpy())
