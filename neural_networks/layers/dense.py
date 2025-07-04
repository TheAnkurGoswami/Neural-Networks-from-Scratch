from typing import Dict, Optional, Tuple, Type # Added Type

import numpy as np
import torch as pt

# Updated import paths
from neural_networks.core.activation_base import Activation # Base class
from neural_networks.activations import Identity, ReLU, Sigmoid, Softmax, Tanh # Specific activations
from neural_networks.backend import ARRAY_TYPE, get_backend
from neural_networks.core.optimizer_base import Optimizer # Base class


# Helper to map activation strings to classes
# This replaces the get_activation_fn previously imported from neural_networks.activations
_activation_map: Dict[str, Type[Activation]] = {
    "identity": Identity,
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "softmax": Softmax,
    "tanh": Tanh,
}

def get_activation_fn(activation_name: Optional[str]) -> Activation:
    """
    Retrieves an instance of an activation function by its name.
    If activation_name is None or "identity", returns an Identity activation instance.
    """
    if activation_name is None:
        activation_name = "identity"

    activation_class = _activation_map.get(activation_name.lower())
    if activation_class is None:
        raise ValueError(f"Unknown activation function: {activation_name}. "
                         f"Available: {list(_activation_map.keys())}")
    return activation_class()


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
        _activation_fn (Activation): Instance of the activation function to be applied.
        add_bias (bool): If True, a bias term is added to the linear transformation.
        _inputs (Optional[ARRAY_TYPE]): Stores the input to the layer during the forward pass,
                                        used for backpropagation.
        _weights (ARRAY_TYPE): The learnable weights matrix of shape (in_features, out_features).
        _bias (Optional[ARRAY_TYPE]): The learnable bias vector of shape (1, out_features),
                                      or None if `add_bias` is False.
        _dw_history (Optional[Dict[str, ARRAY_TYPE]]): Optimizer history for weight gradients.
        _db_history (Optional[Dict[str, ARRAY_TYPE]]): Optimizer history for bias gradients.
        _retain_grad (bool): If True, stores gradients \(dW, dB, dZ\) for debugging.
        _dW (Optional[ARRAY_TYPE]): Gradient of loss w.r.t. weights (if _retain_grad).
        _dB (Optional[ARRAY_TYPE]): Gradient of loss w.r.t. bias (if _retain_grad).
        _dZ (Optional[ARRAY_TYPE]): Gradient of loss w.r.t. linear output Z (if _retain_grad).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Optional[str] = None, # activation is a string name
        add_bias: bool = True,
        retain_grad: bool = False,
    ) -> None:
        """
        Initializes the Dense layer.

        Parameters:
        - in_features (int): Number of input features.
        - out_features (int): Number of output features.
        - activation (Optional[str]): Name of the activation function to use (e.g., "relu", "sigmoid").
                                      Default is None, which means "identity" activation.
        - add_bias (bool): Whether to include a bias term. Default is True.
        - retain_grad (bool): Whether to store gradients for debugging. Default is False.
        """
        self._in_features = in_features
        self._out_features = out_features
        # Get an instance of the activation function
        self._activation_fn: Activation = get_activation_fn(activation)
        self.add_bias = add_bias

        self._inputs: Optional[ARRAY_TYPE] = None
        self._weights: ARRAY_TYPE
        self._bias: Optional[ARRAY_TYPE] # Bias can be None
        self._weights, self._bias = self._build()

        # For moving average based optimizers
        self._dw_history: Optional[Dict[str, ARRAY_TYPE]] = None
        self._db_history: Optional[Dict[str, ARRAY_TYPE]] = None
        self._retain_grad = retain_grad

        if self._retain_grad:
            self._dW: Optional[ARRAY_TYPE] = None
            self._dB: Optional[ARRAY_TYPE] = None
            self._dZ: Optional[ARRAY_TYPE] = None # Gradient before activation

    def _build(self) -> Tuple[ARRAY_TYPE, Optional[ARRAY_TYPE]]:
        r"""
        Initializes the weights and bias for the Dense layer.

        Weights (\(W\)) are initialized using a variation of Xavier/Glorot initialization
        by default, or Kaiming He initialization if the activation function is ReLU.
        Specifically:
            - For ReLU activation: \( W \sim \mathcal{N}(0, \sqrt{2 / \text{in\_features}}) \)
            - For other activations: \( W \sim \mathcal{N}(0, \sqrt{1 / \text{in\_features}}) \)

        Bias (\(b\)), if used, is initialized to zeros.

        Returns:
            Tuple[ARRAY_TYPE, Optional[ARRAY_TYPE]]:
                A tuple containing the initialized weights matrix and the bias vector.
                The bias is None if `add_bias` is False.
        """
        backend, backend_module = get_backend()
        # Kaiming He initialization factor for ReLU-like activations, Xavier/Glorot for others
        # This is a common heuristic.
        if isinstance(self._activation_fn, ReLU):
            # Kaiming He initialization: std = sqrt(2 / in_features)
            std_dev = backend.sqrt(2.0 / self._in_features)
        else:
            # Xavier/Glorot initialization: std = sqrt(1 / in_features) or sqrt(2 / (in_features + out_features))
            # Using sqrt(1/in_features) as a simpler form often found.
            std_dev = backend.sqrt(1.0 / self._in_features)

        if backend_module == "pt":
            self._weights = pt.normal(
                mean=0.0, std=std_dev, size=(self._in_features, self._out_features)
            )
            bias_val = pt.zeros(size=(1, self._out_features)) if self.add_bias else None
        elif backend_module == "np":
            self._weights = np.random.normal(
                loc=0.0,
                scale=std_dev, # Use calculated std_dev
                size=(self._in_features, self._out_features),
            ).astype(np.float32)
            bias_val = np.zeros(shape=(1, self._out_features)).astype(np.float32) if self.add_bias else None
        else:
            raise NotImplementedError(f"Backend {backend_module} not supported for Dense layer initialization.")

        self._bias = bias_val
        return self._weights, self._bias

    def forward(self, inputs: ARRAY_TYPE) -> ARRAY_TYPE:
        r"""
        Performs the forward pass of the Dense layer.

        The operation is \( Z = XW + b \) followed by \( A = \text{activation}(Z) \).
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
            ARRAY_TYPE: The output of the layer after applying the activation function.
                        Shape: (batch_size, out_features).
        """
        backend, backend_module = get_backend()

        # Ensure inputs are in the correct backend format
        if backend_module == "pt":
            if not isinstance(inputs, pt.Tensor):
                self._inputs = pt.tensor(inputs, dtype=pt.float32)
            else:
                # If already a tensor, ensure it's float32.
                # Cloning and detaching might be needed if inputs require grad elsewhere,
                # but for internal layer use, direct use is fine if it's already detached.
                self._inputs = inputs.float()
            # For PyTorch, if inputs might require grad from previous layers,
            # and we need to track graph through this layer's operations on _inputs,
            # then self._inputs = inputs.clone() or just self._inputs = inputs might be fine.
            # The original code had .requires_grad_(True) which is usually for leaf tensors
            # or if you want to break the graph and start a new one from this point.
            # Assuming inputs are outputs of previous layers or raw data.
        elif backend_module == "np":
            if not isinstance(inputs, np.ndarray):
                self._inputs = np.array(inputs, dtype=np.float32)
            else:
                self._inputs = inputs.astype(np.float32) # Ensure float32
        else:
            raise NotImplementedError(f"Backend {backend_module} not supported for Dense layer forward pass.")

        # Linear transformation: Z = XW + b
        linear_output = backend.matmul(self._inputs, self._weights)
        if self.add_bias and self._bias is not None:
            linear_output += self._bias

        # Apply activation function: A = activation(Z)
        activation_output = self._activation_fn.forward(linear_output)
        return activation_output

    def backprop(self, dA: ARRAY_TYPE, optimizer: Optimizer) -> ARRAY_TYPE:
        r"""
        Performs the backpropagation step for the Dense layer.

        Calculates gradients with respect to inputs (\( dX \)), weights (\( dW \)), and bias (\( dB \)),
        and updates weights and bias using the provided optimizer.

        Steps:
        1. Compute \( dZ = dA \odot \text{activation}'(Z) \), where \( Z \) was the input to activation.
           \( \text{activation}'(Z) \) is \( \frac{\partial A}{\partial Z} \).
           \( dA \) is \( \frac{\partial L}{\partial A} \).
           So, \( dZ = \frac{\partial L}{\partial A} \odot \frac{\partial A}{\partial Z} = \frac{\partial L}{\partial Z} \).
        2. Compute gradient for weights: \( dW = X^T dZ \). Averaged over batch if necessary.
        3. Compute gradient for bias: \( dB = \sum dZ \) (sum over batch dimension).
        4. Compute gradient for input to this layer: \( dX = dZ W^T \).
        5. Update weights: \( W = W - \text{optimizer\_step}(dW) \).
        6. Update bias: \( b = b - \text{optimizer\_step}(dB) \).

        Args:
            dA (ARRAY_TYPE): Gradient of the loss with respect to the activation output of this layer.
                             Shape: (batch_size, out_features).
            optimizer (Optimizer): Optimizer to use for updating the weights and biases.

        Returns:
            ARRAY_TYPE: Gradient of the loss with respect to the inputs of this layer (\( dX \)).
                        Shape: (batch_size, in_features).
        """
        if self._inputs is None:
            raise ValueError("Forward pass must be called before backprop.")

        backend, backend_module = get_backend()

        # 1. Compute dZ = dA * activation_derivative(Z_linear_output)
        # The activation_fn's backprop method should implement dA * derivative_of_activation_input
        dZ = self._activation_fn.backprop(dA) # dZ has shape (batch_size, out_features)

        batch_size = self._inputs.shape[0]

        # 2. Compute dW = (1/batch_size) * X^T dZ
        # Transpose self._inputs: (in_features, batch_size)
        # dZ: (batch_size, out_features)
        # dW: (in_features, out_features)
        # Note: some frameworks average gradient over batch size here, others in optimizer or loss.
        # Assuming gradients are per-sample summed up, then optimizer might average.
        # The original code does not average here. Let's stick to that for now.
        # If inputs can be 3D (e.g. from attention), sum over the sequence dimension if not batch.
        # For typical Dense, inputs are (batch, features).
        inputs_T = self._inputs.transpose(-1, -2) if self._inputs.ndim > 1 else self._inputs.reshape(-1,1) # Handle 1D input for a single feature
        if self._inputs.ndim == 1: # If input was a single sample (features_dim,)
             inputs_T = self._inputs.reshape(-1,1) # (in_features, 1)
             if dZ.ndim == 1 : dZ_reshaped = dZ.reshape(1,-1) # (1, out_features)
             else: dZ_reshaped = dZ
             dW = backend.matmul(inputs_T, dZ_reshaped)
        else: # Batch input
            dW = backend.matmul(inputs_T, dZ)

        # If dW is accumulated over a batch and needs averaging:
        # dW = dW / batch_size
        # Current implementation follows original: no averaging of dW/dB here.

        # 5. Update weights
        dw_update, self._dw_history = optimizer.optimize(self._dw_history, dW)
        self._weights -= dw_update

        # 3. Compute dB = (1/batch_size) * sum(dZ, axis=0)
        if self.add_bias and self._bias is not None:
            # Sum dZ over the batch dimension (axis 0)
            # dZ shape is (batch_size, out_features)
            # dB shape should be (1, out_features) to match bias shape
            if backend_module == "pt":
                dB = backend.sum(dZ, dim=0, keepdim=True)
            elif backend_module == "np":
                dB = backend.sum(dZ, axis=0, keepdims=True)
            else:
                raise NotImplementedError(f"Backend {backend_module} not supported for bias gradient sum.")
            # dB = dB / batch_size # If averaging

            # 6. Update bias
            db_update, self._db_history = optimizer.optimize(self._db_history, dB)
            self._bias -= db_update
        else:
            dB = None # For retain_grad

        # 4. Compute dX = dZ @ W^T
        # dZ: (batch_size, out_features)
        # self._weights.T: (out_features, in_features)
        # dX: (batch_size, in_features)
        dX = backend.matmul(dZ, self._weights.T)

        if self._retain_grad:
            self._dW = dW
            self._dB = dB if self.add_bias else None
            self._dZ = dZ # Gradient before activation (output of activation.backprop)

        return dX

    # Properties to access weights and biases if needed
    @property
    def weights(self) -> ARRAY_TYPE:
        return self._weights

    @weights.setter
    def weights(self, value: ARRAY_TYPE) -> None:
        self._weights = value

    @property
    def bias(self) -> Optional[ARRAY_TYPE]:
        return self._bias

    @bias.setter
    def bias(self, value: Optional[ARRAY_TYPE]) -> None:
        self._bias = value

    # Properties for retained gradients if used
    @property
    def dW(self) -> Optional[ARRAY_TYPE]:
        if not self._retain_grad:
            raise AttributeError("Gradients are not retained. Set retain_grad=True.")
        return self._dW

    @property
    def dB(self) -> Optional[ARRAY_TYPE]:
        if not self._retain_grad:
            raise AttributeError("Gradients are not retained. Set retain_grad=True.")
        return self._dB

    @property
    def dZ(self) -> Optional[ARRAY_TYPE]:
        if not self._retain_grad:
            raise AttributeError("Gradients are not retained. Set retain_grad=True.")
        return self._dZ
