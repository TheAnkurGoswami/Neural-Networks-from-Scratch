from typing import Any, Dict

import numpy as np
import pytest
import tensorflow as tf

from neural_networks.losses import RMSELoss
from neural_networks.nn import Dense
from neural_networks.optimizers import get_optimizer


@pytest.mark.parametrize("optimizer_str", ["sgd"])
@pytest.mark.parametrize("kwargs", [{"learning_rate": 0.001, "momentum": 0.9}])
def test_n_hidden_layer_simple_nn(
        optimizer_str: str, kwargs: Dict[str, Any]) -> None:
    epochs = 10
    np.random.seed(78)

    x = np.random.randint(low=0, high=10, size=(1, 5))
    y = np.random.randint(low=0, high=10, size=(1, ))

    hidden_layers_size = [3, 12, 7]
    layers = [x.shape[1]] + hidden_layers_size + [1]
    n_layers = len(layers) - 1
    dense_layers = []
    tf_weights_list = []
    tf_biases_list = []
    optimizer = get_optimizer(optimizer_str)(**kwargs)
    for idx in range(n_layers):
        dense = Dense(in_features=layers[idx], out_features=layers[idx+1])
        w = dense._weights.copy()
        b = dense._bias.copy()
        dense_layers.append(dense)

        w_tf = tf.Variable(w.astype(np.float32))
        b_tf = tf.Variable(b, dtype=tf.float32)
        tf_weights_list.append(w_tf)
        tf_biases_list.append(b_tf)

    x_tf = tf.constant(x.astype(np.float32))
    y_tf = tf.constant(y.astype(np.float32))

    loss = RMSELoss()
    optimizer_tf = tf.keras.optimizers.SGD(**kwargs)
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
        optimizer_tf.apply_gradients(zip(grads, trainable_variables))

        for idx in range(n_layers):
            # print(dense_layers[idx]._weights, tf_weights_list[idx])
            # for i in range(dense_layers[idx]._weights.shape[0]):
            #     print(dense_layers[idx]._weights[i], tf_weights_list[idx][i])
            #     print(np.allclose(
            #         dense_layers[idx]._weights[i], tf_weights_list[idx][i]))
            assert np.allclose(
                dense_layers[idx]._weights, tf_weights_list[idx])
            assert np.allclose(dense_layers[idx]._bias, tf_biases_list[idx])
        print(cost_nn, cost_tf)
        assert np.allclose(cost_nn, cost_tf)
