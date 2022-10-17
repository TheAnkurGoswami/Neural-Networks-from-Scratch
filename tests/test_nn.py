import numpy as np
import tensorflow as tf
from neural_networks.nn import Dense
from neural_networks.losses import MSELoss


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
    costs_nn = []
    costs_tf = []
    for _ in range(epochs):
        # Our neural network
        y_pred = dense.forward(inputs=x)
        cost_nn = loss.forward(y_pred=y_pred, y_true=y)
        costs_nn.append(cost_nn)
        dL = loss.backprop()
        dense.backprop(dL, learning_rate=learning_rate)

        # Tensorflow neural network
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        loss_fn = tf.keras.losses.MeanSquaredError()
        with tf.GradientTape() as tape:
            y_hat = tf.matmul(X, W) + b
            cost_tf = loss_fn(y_hat, Y)
            costs_tf.append(cost_tf)
        trainable_variables = [W, b]
        grads = tape.gradient(cost_tf, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))
        
        assert np.allclose(dense._weights, W)
        assert np.allclose(dense._bias, b)
        assert np.allclose(cost_nn, cost_tf)