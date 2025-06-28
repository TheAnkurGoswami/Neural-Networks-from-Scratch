from typing import Dict, Optional, Tuple

import numpy as np
import torch as pt

from neural_networks.activations import get_activation_fn
from neural_networks.backend import ARRAY_TYPE, get_backend
from neural_networks.optimizers import Optimizer


class Dense:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Optional[str] = None,
        add_bias: bool = True,
        retain_grad: bool = False,
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
        self.add_bias = add_bias
        self._inputs: Optional[ARRAY_TYPE] = None
        self._weights, self._bias = self._build()

        # For moving average based optimizers
        self._dw_history: Optional[Dict[str, ARRAY_TYPE]] = None
        self._db_history: Optional[Dict[str, ARRAY_TYPE]] = None
        self._retain_grad = retain_grad  # For debugging purpose

        if self._retain_grad:
            self._dW: Optional[ARRAY_TYPE] = None
            self._dB: Optional[ARRAY_TYPE] = None
            self._dZ: Optional[ARRAY_TYPE] = None

    def _build(self) -> Tuple[ARRAY_TYPE, ARRAY_TYPE]:
        """
        Builds the weights and bias for the layer.

        Returns:
        - Tuple[ARRAY_TYPE, ARRAY_TYPE]: Initialized weights and bias.
        """
        _, backend_module = get_backend()
        if backend_module == "pt":
            weights = pt.normal(
                mean=0.0, std=1.0, size=(self._in_features, self._out_features)
            )
            bias = pt.zeros(size=(1, self._out_features))
        elif backend_module == "np":
            weights = np.random.normal(
                loc=0.0,
                scale=1.0,
                size=(self._in_features, self._out_features),
            ).astype(np.float32)
            bias = np.zeros(shape=(1, self._out_features)).astype(np.float32)
        if not self.add_bias:
            bias = None
        return weights, bias

    def forward(self, inputs: ARRAY_TYPE) -> ARRAY_TYPE:
        """
        Performs the forward pass of the neural network layer.

        Parameters:
        - inputs (ARRAY_TYPE): The input data to the layer.
            Shape: (n_inputs, self._in_features)

        Returns:
        - ARRAY_TYPE: The output of the layer after applying the activation
            function. Shape: (n_inputs, self._out_features)
        """
        backend, backend_module = get_backend()
        if backend_module == "pt":
            if isinstance(inputs, pt.Tensor):
                self._inputs = inputs.clone().detach().requires_grad_(True)
            else:
                self._inputs = pt.tensor(inputs, dtype=pt.float32)
        elif backend_module == "np":
            self._inputs = np.array(inputs)
        result = backend.matmul(self._inputs, self._weights)
        if self.add_bias:
            result += self._bias
        activation = self._activation.forward(result)
        return activation

    def backprop(self, dA: ARRAY_TYPE, optimizer: Optimizer) -> ARRAY_TYPE:
        """
        Performs the backpropagation step for the layer.

        Parameters:
        - dA (ARRAY_TYPE): Gradient of the loss with respect to the activation.
            Shape: (n_inputs, self._out_features)
        - optimizer (Optimizer): Optimizer to use for updating the weights and
            biases.

        Returns:
        - ARRAY_TYPE: Gradient of the loss with respect to the inputs.
                    Shape: (n_inputs, self._in_features)
        """
        assert self._inputs is not None
        backend, backend_module = get_backend()
        dZ = self._activation.backprop(dA)
        # print(self._inputs.shape, dZ.shape)
        dW = backend.matmul(self._inputs.transpose(-1, -2), dZ)
        if dW.ndim == 3:
            dW = dW.sum(axis=0)
        # print("dW", dW)
        dw_change, self._dw_history = optimizer.optimize(self._dw_history, dW)
        dX = backend.matmul(dZ, self._weights.T)
        self._weights -= dw_change

        if self.add_bias:
            if backend_module == "pt":
                dB = backend.sum(dZ, dim=0, keepdim=True)
            elif backend_module == "np":
                dB = backend.sum(dZ, axis=0, keepdims=True)
            if dB.ndim == 3:
                dB = dB.sum(axis=1)
            # print("dB", dB)

            db_change, self._db_history = optimizer.optimize(
                self._db_history, dB
            )
            # Parametric updates
            self._bias -= db_change
        else:
            dB = None

        if self._retain_grad:
            self._dW = dW
            self._dB = dB
            self._dZ = dZ
        return dX
