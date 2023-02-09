from typing import Any, Dict

import numpy as np
import pytest
import tensorflow as tf
import torch

from neural_networks.losses import RMSELoss
from neural_networks.nn import Dense
from neural_networks.optimizers import get_optimizer
from utils import check_closeness

TF_OPTIM_MAP = {
    "sgd": tf.keras.optimizers.SGD,
    "rmsprop": tf.keras.optimizers.RMSprop,
    "adam": tf.keras.optimizers.Adam,
}

TORCH_OPTIM_MAP = {
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "adam": torch.optim.Adam,
}


@pytest.mark.parametrize(
    "optimizer_str, kwargs",
    [
        ("sgd", {"learning_rate": 0.001, "momentum": 0.9}),
        ("rmsprop", {"learning_rate": 0.001, "rho": 0.9, "epsilon": 1e-07}),
        ("adam", {
            "learning_rate": 0.001,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-07})
    ])
def test_optimizer(optimizer_str: str, kwargs: Dict[str, Any]) -> None:
    epochs = 10
    np.random.seed(65)

    x = np.random.randint(low=0, high=10, size=(1, 5))
    y = np.random.randint(low=0, high=10, size=(1, 1))

    hidden_layers_size = [3, 12, 7]
    layers = [x.shape[1]] + hidden_layers_size + [1]
    n_layers = len(layers) - 1
    dense_layers = []
    tf_weights_list = []
    tf_biases_list = []
    torch_weights_list = []
    torch_biases_list = []
    optimizer = get_optimizer(optimizer_str)(**kwargs)
    for idx in range(n_layers):
        dense = Dense(in_features=layers[idx], out_features=layers[idx+1])
        w = dense._weights.copy()
        b = dense._bias.copy()
        dense_layers.append(dense)

        w_tf = tf.Variable(w.astype(np.float32))
        b_tf = tf.Variable(b.astype(np.float32))
        tf_weights_list.append(w_tf)
        tf_biases_list.append(b_tf)

        w_torch = torch.tensor(w, requires_grad=True, dtype=torch.float32)
        b_torch = torch.tensor(b, requires_grad=True)
        torch_weights_list.append(w_torch)
        torch_biases_list.append(b_torch)

    x_tf = tf.constant(x.astype(np.float32))
    y_tf = tf.constant(y.astype(np.float32))
    x_torch = torch.tensor(x.astype(np.float32), requires_grad=True)
    y_torch = torch.tensor(y.astype(np.float32))

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
        params=[*torch_weights_list, *torch_biases_list],
        **new_kwargs)
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
                feed_in = output
            cost_tf = tf.sqrt(tf.losses.mean_squared_error(output, y_tf))
        trainable_variables = [*tf_weights_list, *tf_biases_list]
        grads = tape.gradient(cost_tf, trainable_variables)
        optimizer_tf.apply_gradients(zip(grads, trainable_variables))

        # Pytorch neural network
        all_dX = []
        all_dZ = []
        feed_in_torch = x_torch
        for idx in range(n_layers):
            optimizer_torch.zero_grad()
            output_torch = torch.matmul(
                feed_in_torch,
                torch_weights_list[idx]) + torch_biases_list[idx]
            feed_in_torch = output_torch
        loss_torch = torch.nn.MSELoss()
        loss_torch_fn = torch.sqrt(loss_torch(output_torch, y_torch))
        loss_torch_fn.backward()
        optimizer_torch.step()

        for idx in range(n_layers):
            assert check_closeness(
                dense_layers[idx]._weights,
                tf_weights_list[idx],
                double_check=True)
            assert check_closeness(
                dense_layers[idx]._weights,
                torch_weights_list[idx].detach().numpy(),
                double_check=True)

            assert check_closeness(
                dense_layers[idx]._bias,
                tf_biases_list[idx],
                double_check=True)
            assert check_closeness(
                dense_layers[idx]._bias,
                torch_biases_list[idx].detach().numpy(),
                double_check=True)
        assert check_closeness(cost_nn, cost_tf)
        assert check_closeness(cost_nn, loss_torch_fn.item())
