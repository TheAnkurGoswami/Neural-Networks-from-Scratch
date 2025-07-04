from typing import Dict, Optional, Tuple

import numpy as np
import torch as pt

from neural_networks.activations import (
    get_activation_fn,
)

# Updated import paths
from neural_networks.backend import ARRAY_TYPE, get_backend
from neural_networks.optimizers.base import Optimizer  # Base class


class Dense:
    """
    A fully connected neural network layer (Linear layer).

    This layer performs a linear transformation of the input data \(X\) using
    a weights matrix \(W\) and an optional bias vector \(b\):
    \( Z = XW + b \)
    The result \(Z\) is then passed through an activation function \(f\):
    \( A = f(Z) \)

    Attributes:
        _in_features (int): Number of input features.
        _out_features (int): Number of output features (neurons in this layer).
        _activation_fn (Activation): Instance of the activation function to be
            applied.
        add_bias (bool): If True, a bias term is added to the linear
            transformation.
        _inputs (Optional[ARRAY_TYPE]): Stores the input to the layer during
            the forward pass, used for backpropagation.
        _weights (ARRAY_TYPE): The learnable weights matrix of shape
            (in_features, out_features).
        _bias (Optional[ARRAY_TYPE]): The learnable bias vector of shape
            (1, out_features), or None if `add_bias` is False.
        _dw_history (Optional[Dict[str, ARRAY_TYPE]]): Optimizer history for
            weight gradients.
        _db_history (Optional[Dict[str, ARRAY_TYPE]]): Optimizer history for
            bias gradients.
        _retain_grad (bool): If True, stores gradients \(dW, dB, dZ\) for
            debugging.
        _dW (Optional[ARRAY_TYPE]): Gradient of loss w.r.t. weights
            (if _retain_grad).
        _dB (Optional[ARRAY_TYPE]): Gradient of loss w.r.t. bias
            (if _retain_grad).
        _dZ (Optional[ARRAY_TYPE]): Gradient of loss w.r.t. linear output Z
            (if _retain_grad).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Optional[str] = None,  # activation is a string name
        add_bias: bool = True,
        retain_grad: bool = False,
    ) -> None:
        """
        Initializes the Dense layer.

        Parameters:
        - in_features (int): Number of input features.
        - out_features (int): Number of output features.
        - activation (Optional[str]): Name of the activation function to use
            (e.g., "relu", "sigmoid"). Default is None, which means "identity"
            activation.
        - add_bias (bool): Whether to include a bias term. Default is True.
        - retain_grad (bool): Whether to store gradients for debugging.
            Default is False.
        """
        self._in_features = in_features
        self._out_features = out_features
        # Get an instance of the activation function
        self._activation = get_activation_fn(activation)()
        self.add_bias = add_bias

        self._inputs: Optional[ARRAY_TYPE] = None
        self._weights, self._bias = self._build()

        # For moving average based optimizers
        self._dw_history: Optional[Dict[str, ARRAY_TYPE]] = None
        self._db_history: Optional[Dict[str, ARRAY_TYPE]] = None
        self._retain_grad = retain_grad

        if self._retain_grad:
            self._dW: Optional[ARRAY_TYPE] = None
            self._dB: Optional[ARRAY_TYPE] = None
            self._dZ: Optional[ARRAY_TYPE] = None  # Gradient before activation

    def _build(self) -> Tuple[ARRAY_TYPE, Optional[ARRAY_TYPE]]:
        r"""
        Initializes the weights and bias for the Dense layer.
        Bias (\(b\)), if used, is initialized to zeros.

        Returns:
            Tuple[ARRAY_TYPE, Optional[ARRAY_TYPE]]:
                A tuple containing the initialized weights matrix and the bias
                vector. The bias is None if `add_bias` is False.
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
        r"""
        Performs the forward pass of the Dense layer.

        The operation is \( Z = XW + b \) followed by
        \( A = \text{activation}(Z) \).
        Where:
         - \( X \) is the input data.
         - \( W \) is the weights matrix.
         - \( b \) is the bias vector (if add_bias is True).
         - \( Z \) is the linear output.
         - \( A \) is the activation output.

        Args:
            inputs (ARRAY_TYPE): The input data to the layer.
                Shape: (batch_size, in_features).

        Returns:
            ARRAY_TYPE: The output of the layer after applying the activation
            function. Shape: (batch_size, out_features).
        """
        backend, backend_module = get_backend()

        # Ensure inputs are in the correct backend format
        if backend_module == "pt":
            if isinstance(inputs, pt.Tensor):
                self._inputs = inputs.clone().detach().requires_grad_(True)
            else:
                self._inputs = pt.tensor(inputs, dtype=pt.float32)
        elif backend_module == "np":
            self._inputs = np.array(inputs)
        else:
            raise NotImplementedError(
                f"Backend {backend_module} not supported for Dense layer "
                "forward pass."
            )

        # Linear transformation: Z = XW + b
        result = backend.matmul(self._inputs, self._weights)
        if self.add_bias and self._bias is not None:
            result += self._bias
        # Apply activation function: A = activation(Z)
        activation = self._activation.forward(result)
        return activation

    def backprop(self, dA: ARRAY_TYPE, optimizer: Optimizer) -> ARRAY_TYPE:
        r"""
        Performs the backpropagation step for the Dense layer.

        Calculates gradients w.r.t inputs (\( dX \)), weights (\( dW \)), and
        bias (\( dB \)), and updates weights and bias using the provided
        optimizer.

        Steps:
        1. Compute \( dZ = dA \odot \text{activation}'(Z) \), where \( Z \)
            was the input to activation.
            \( \text{activation}'(Z) \) is \( \frac{\partial A}{\partial Z} \).
            \( dA \) is \( \frac{\partial L}{\partial A} \).
            So, \( dZ = \frac{\partial L}{\partial A}
            \odot \frac{\partial A}{\partial Z} =
            \frac{\partial L}{\partial Z} \).
        2. Compute gradient for weights: \( dW = X^T dZ \).
            Averaged over batch if necessary.
        3. Compute gradient for bias: \( dB = \sum dZ \)
            (sum over batch dimension).
        4. Compute gradient for input to this layer: \( dX = dZ W^T \).
        5. Update weights: \( W = W - \text{optimizer\_step}(dW) \).
        6. Update bias: \( b = b - \text{optimizer\_step}(dB) \).

        Args:
            dA (ARRAY_TYPE): Gradient of the loss w.r.t the activation output
                of this layer. Shape: (batch_size, out_features).
            optimizer (Optimizer): Optimizer to use for updating the weights
                and biases.

        Returns:
            ARRAY_TYPE: Gradient of the loss w.r.t the inputs of this layer
                (\( dX \)). Shape: (batch_size, in_features).
        """
        if self._inputs is None:
            raise ValueError("Forward pass must be called before backprop.")

        backend, backend_module = get_backend()

        # 1. Compute dZ = dA * activation_derivative(Z_linear_output)
        # The activation_fn's backprop method should implement dA * derivative
        # of_activation_input
        dZ = self._activation.backprop(dA)
        # dZ has shape (batch_size, out_features)

        # 2. Compute dW = X^T dZ
        dW = backend.matmul(self._inputs.transpose(-1, -2), dZ)
        if dW.ndim == 3:
            dW = dW.sum(axis=0)

        # 3. Compute dB = sum(dZ, axis=0)
        if self.add_bias and self._bias is not None:
            # Sum dZ over the batch dimension (axis 0)
            # dZ shape is (batch_size, out_features)
            # dB shape should be (1, out_features) to match bias shape
            if backend_module == "pt":
                dB = backend.sum(dZ, dim=0, keepdim=True)
            elif backend_module == "np":
                dB = backend.sum(dZ, axis=0, keepdims=True)
            else:
                raise NotImplementedError(
                    f"Backend {backend_module} not supported for bias "
                    "gradient sum."
                )
            if dB.ndim == 3:
                dB = dB.sum(axis=1)

            # 6. Update bias
            db_change, self._db_history = optimizer.optimize(
                self._db_history, dB
            )
            # Parametric updates
            self._bias -= db_change
        else:
            dB = None

        # 4. Compute dX = dZ @ W^T
        # dZ: (batch_size, out_features)
        # self._weights.T: (out_features, in_features)
        # dX: (batch_size, in_features)
        dX = backend.matmul(dZ, self._weights.T)

        # 5. Update weights
        dw_change, self._dw_history = optimizer.optimize(self._dw_history, dW)
        self._weights -= dw_change

        if self._retain_grad:
            self._dW = dW
            self._dB = dB if self.add_bias else None
            self._dZ = dZ  # Gradient before activation
        return dX
