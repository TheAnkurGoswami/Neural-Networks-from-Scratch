from typing import Dict, Optional, Tuple

import numpy as np

from neural_networks.activations import get_activation_fn
from neural_networks.optimizers import Optimizer


class Dense:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Optional[str] = None,
    ) -> None:
        """
        Initializes the Dense layer.

        Parameters:
        - in_features (int): Number of input features.
        - out_features (int): Number of output features.
        - activation (Optional[str]): Activation function to use.
            Default is None.
        """
        self._in_features = in_features
        self._out_features = out_features
        self._activation = get_activation_fn(activation)()
        self._inputs: Optional[np.ndarray] = None
        self._weights, self._bias = self._build()

        # For moving average based optimizers
        self._dw_history: Optional[Dict[str, np.ndarray]] = None
        self._db_history: Optional[Dict[str, np.ndarray]] = None

    def _build(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds the weights and bias for the layer.

        Returns:
        - Tuple[np.ndarray, np.ndarray]: Initialized weights and bias.
        """
        weights = np.random.normal(
            loc=0.0, scale=1.0, size=(self._in_features, self._out_features)
        ).astype(np.float32)
        bias = np.zeros(shape=(1, self._out_features)).astype(np.float32)
        return weights, bias

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the neural network layer.

        Parameters:
        - inputs (np.ndarray): The input data to the layer.
            Shape: (n_inputs, self._in_features)

        Returns:
        - np.ndarray: The output of the layer after applying the activation
            function. Shape: (n_inputs, self._out_features)
        """
        self._inputs = inputs
        result = np.matmul(inputs, self._weights) + self._bias
        activation = self._activation.forward(result)
        return activation

    def backprop(self, dA: np.ndarray, optimizer: Optimizer) -> np.ndarray:
        """
        Performs the backpropagation step for the layer.

        Parameters:
        - dA (np.ndarray): Gradient of the loss with respect to the activation.
            Shape: (n_inputs, self._out_features)
        - optimizer (Optimizer): Optimizer to use for updating the weights and
            biases.

        Returns:
        - np.ndarray: Gradient of the loss with respect to the inputs.
                    Shape: (n_inputs, self._in_features)
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

        # Parametric updates
        self._weights -= dw_change
        self._bias -= db_change

        return dX
