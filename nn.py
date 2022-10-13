import numpy as np

from activations import get_activation_fn

class Dense:
    def __init__(self, n_inputs, n_outputs, activation=None) -> None:
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._activation = activation
        self._inputs = None
        self._weights = None
        self._bias = None
        self._build()
  
    def _build(self) -> None:
        self._weights = np.random.normal(loc=0.0, scale=1.0, size=(self._n_outputs, self._n_inputs))
        self._bias = np.zeros(shape=(self._n_outputs, 1))
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        inputs.shape = (self._n_inputs, input_batch_size)
        result.shape = (self._n_outputs, input_batch_size)
        activation.shape = (self._n_outputs, input_batch_size)
        """
        self._inputs = inputs
        # z = w.x + b
        result = np.matmul(self._weights, inputs) + self._bias
        activation_fn = get_activation_fn(self._activation)
        activation = activation_fn.forward(result)
        return activation

    def backprop(self, dA: np.ndarray, learning_rate: int) -> np.ndarray:
        """
        dA is short notation for dL/da (change in loss w.r.t change in activation).
        dZ is short notation for dL/dz.

        z = w.x + b
        differentiating the equation w.r.t w, we get: 
        dz/dw = x

        Now, (dz/dw)(dL/dz) = x.(dL/dz)
        dL/dw = x.(dL/dz)
        or
        dW = x.dZ

        Similarly, differentiating the equation w.r.t x & b separately, we get:

        dX = w.dZ
        dB = dZ

        """
        activation_fn = get_activation_fn(self._activation)
        dZ = activation_fn.backprop(dA)
        dW = np.matmul(self._inputs.T, dZ)
        dB = np.sum(dZ, axis=-1)
        dX = np.matmul(self._weights.T, dZ)

        # W = W - alpha * dW
        self._weights -= learning_rate * dW
        # B = B - alpha * dB
        self._bias -= learning_rate * dB

        return dX