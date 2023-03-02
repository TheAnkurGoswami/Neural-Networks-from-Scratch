from typing import Dict, Optional, Tuple

import numpy as np

from neural_networks.activations import get_activation_fn
from neural_networks.optimizers import Optimizer


class Dense:
    def __init__(
            self,
            in_features: int,
            out_features: int,
            activation: Optional[str] = None) -> None:
        self._in_features = in_features
        self._out_features = out_features
        self._activation = get_activation_fn(activation)()
        self._inputs: Optional[np.ndarray] = None
        self._weights, self._bias = self._build()
        # For moving average based optimizers
        self._dw_history: Optional[Dict[str, np.ndarray]] = None
        self._db_history: Optional[Dict[str, np.ndarray]] = None

    def _build(self) -> Tuple[np.ndarray, np.ndarray]:
        weights = np.random.normal(
            loc=0.0, scale=1.0, size=(self._in_features, self._out_features))
        weights = weights.astype(np.float32)
        bias = np.zeros(shape=(1, self._out_features)).astype(np.float32)
        return weights, bias

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        inputs.shape = (n_inputs, self._in_features)
        result.shape = (n_inputs, self._out_features)
        activation.shape = (n_inputs, self._out_features)
        """
        self._inputs = inputs
        # z = x.w + b
        print("x", inputs.shape)
        print("w", self._weights.shape)
        print("b", self._bias.shape)
        result = np.matmul(inputs, self._weights) + self._bias
        print("z", result.shape)
        activation = self._activation.forward(result)
        print("a", activation.shape)
        return activation

    def backprop(self, dA: np.ndarray, optimizer: Optimizer) -> np.ndarray:
        """
        dA is short notation for dL/da (change in loss w.r.t change in
        activation).
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
        assert self._inputs is not None
        dZ = self._activation.backprop(dA)
        print("dZ", dZ.shape)
        dW = np.matmul(self._inputs.T, dZ) #/ dZ.shape[0]
        dB = np.matmul(np.ones((1, dZ.shape[0])), dZ)  # Sum of all elements of dZ along batch
        dB = np.sum(dZ, axis=0)
        dX = np.matmul(dZ, self._weights.T)
        print("dB", dB.shape, dB)
        print("dW", dW.shape)
        dw_change, self._dw_history = optimizer.optimize(self._dw_history, dW)
        db_change, self._db_history = optimizer.optimize(self._db_history, dB)

        # Parametric updations
        print("orig db", self._bias.shape)
        print("change", db_change.shape)
        self._weights -= dw_change
        self._bias -= db_change

        return dX
