import numpy as np

from neural_networks.activations import get_activation_fn

class Dense:
    def __init__(self, in_features, out_features, activation=None) -> None:
        self._in_features = in_features
        self._out_features = out_features
        self._activation = activation
        self._inputs = None
        self._weights = None
        self._bias = None
        self._build()
  
    def _build(self) -> None:
        self._weights = np.random.normal(loc=0.0, scale=1.0, size=(self._in_features, self._out_features))
        self._weights = self._weights.astype(np.float32)
        self._bias = np.zeros(shape=(1, self._out_features)).astype(np.float32)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        inputs.shape = (n_inputs, self._in_features)
        result.shape = (n_inputs, self._out_features)
        activation.shape = (n_inputs, self._out_features)
        """
        self._inputs = inputs
        # z = x.w + b
        result = np.matmul(inputs, self._weights) + self._bias
        activation_fn = get_activation_fn(self._activation)
        activation = activation_fn.forward(result)
        return activation

    def backprop(self, dA: np.ndarray, learning_rate: float) -> np.ndarray:
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
        dB = dZ
        dX = np.matmul(dZ, self._weights.T)

        # W = W - alpha * dW
        self._weights -= learning_rate * dW
        # B = B - alpha * dB
        self._bias -= learning_rate * dB

        return dX