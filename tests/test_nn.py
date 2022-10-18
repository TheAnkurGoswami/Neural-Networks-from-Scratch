from typing import List
import numpy as np
import pytest
import tensorflow as tf
from neural_networks.nn import Dense
from neural_networks.losses import MSELoss, RMSELoss


def test_no_hidden_layer_simple_nn():
    epochs = 10
    learning_rate = 0.01
    np.random.seed(100)
    
    x = np.random.randint(low=0, high=10, size=(1, 5))
    y = np.random.randint(low=0, high=10, size=(1, ))

    dense = Dense(in_features=x.shape[1], out_features=1)
    w = dense._weights.copy()
    b = dense._bias.copy()

    X = tf.constant(x.astype(np.float32))
    Y = tf.constant(y.astype(np.float32))
    W = tf.Variable(w.astype(np.float32))
    b = tf.Variable(b, dtype=tf.float32)

    loss = MSELoss()
    for _ in range(epochs):
        # Our neural network
        y_pred = dense.forward(inputs=x)
        cost_nn = loss.forward(y_pred=y_pred, y_true=y)
        dL = loss.backprop()
        dense.backprop(dL, learning_rate=learning_rate)

        # Tensorflow neural network
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        loss_fn = tf.keras.losses.MeanSquaredError()
        with tf.GradientTape() as tape:
            y_hat = tf.matmul(X, W) + b
            cost_tf = loss_fn(y_hat, Y)
        trainable_variables = [W, b]
        grads = tape.gradient(cost_tf, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))
        
        assert np.allclose(dense._weights, W)
        assert np.allclose(dense._bias, b)
        assert np.allclose(cost_nn, cost_tf)


@pytest.mark.parametrize("hidden_layers_size", [[5], [2, 3], [6, 4, 10]])
def test_n_hidden_layer_simple_nn(hidden_layers_size: List[int]):
    epochs = 10
    learning_rate = 0.001
    np.random.seed(100)
    
    x = np.random.randint(low=0, high=10, size=(1, 5))
    y = np.random.randint(low=0, high=10, size=(1, ))

    layers = [x.shape[1]] + hidden_layers_size + [1]
    n_layers = len(layers) - 1
    dense_layers = []
    tf_weights_list = []
    tf_biases_list = []

    for i in range(n_layers):
        dense = Dense(in_features=layers[i], out_features=layers[i+1])
        w = dense._weights.copy()
        b = dense._bias.copy()
        dense_layers.append(dense)

        W = tf.Variable(w.astype(np.float32))
        b = tf.Variable(b, dtype=tf.float32)
        tf_weights_list.append(W)
        tf_biases_list.append(b)
    print([weight.shape for weight in tf_weights_list])

    X = tf.constant(x.astype(np.float32))
    Y = tf.constant(y.astype(np.float32))

    loss = RMSELoss()
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    for _ in range(epochs):
        # Our neural network
        feed_in = x
        for i in range(n_layers):
            output = dense_layers[i].forward(inputs=feed_in)
            feed_in = output
        cost_nn = loss.forward(y_pred=output, y_true=y)
        dL = loss.backprop()
        derivative = dL
        for i in range(n_layers - 1, -1, -1):
            derivative = dense_layers[i].backprop(derivative, learning_rate=learning_rate)

        # Tensorflow neural network
        feed_in = X
        with tf.GradientTape() as tape:
            for i in range(n_layers):
                output = tf.matmul(feed_in, tf_weights_list[i]) + tf_biases_list[i]
                feed_in = output
            cost_tf = tf.sqrt(tf.losses.mean_squared_error(output, Y))
        trainable_variables = [*tf_weights_list, *tf_biases_list]
        grads = tape.gradient(cost_tf, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))
        
        for i in range(n_layers):
            assert np.allclose(dense_layers[i]._weights, tf_weights_list[i])
            print(dense_layers[i]._bias, tf_biases_list[i])
            assert np.allclose(dense_layers[i]._bias, tf_biases_list[i])
        assert np.allclose(cost_nn, cost_tf)