from typing import Optional

from neural_networks.backend import ARRAY_TYPE, get_backend
from neural_networks.layers import Dense  # Updated: Import Dense from layers
from neural_networks.optimizers.base import (
    Optimizer,
)  # Updated: Import Optimizer base


class Projection(Dense):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        add_bias: bool = True,
        activation: Optional[
            str
        ] = None,  # Projection usually doesn't have activation
        **kwargs
    ) -> None:
        r"""
        Initializes a Projection layer, which is a specialized Dense layer.
        It performs a linear transformation: \( Z = XW + b \).
        This class is similar to `Dense` but is often used in attention
        mechanisms where the input \(X\) might have a sequence dimension, and
        gradients \(dW, dB\) might be summed differently across the batch or
        sequence. The primary difference from a standard `Dense` layer here is
        how bias is shaped and potentially how gradients are handled if inputs
        are 3D (batch, seq, features).

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            add_bias (bool, optional): If True, adds a learnable bias.
                Defaults to True.
            activation (Optional[str], optional): Activation function name.
                For typical projection layers in attention, this is usually
                None (identity). Defaults to None.
            **kwargs: Additional arguments passed to the parent `Dense` class
                constructor.
        """
        # Projections in attention typically don't have an activation function
        # applied directly within them. The output of projections (Q, K, V) are
        # then used in dot products, softmax, etc. So, we pass activation=None
        # (or "identity") to the Dense layer.
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            add_bias=add_bias,
            activation=activation,  # Pass along, defaults to None/identity
            **kwargs
        )

        # Reshape bias to (1, 1, out_features) for broadcasting with 3D inputs
        # (batch, seq, features)
        # This is a common pattern in attention mechanisms.
        # The Dense layer initializes bias as (1, out_features).
        if self.add_bias and self._bias is not None:
            backend, _ = get_backend()
            self._bias = backend.reshape(self._bias, (1, 1, out_features))

    def backprop(self, dZ: ARRAY_TYPE, optimizer: Optimizer) -> ARRAY_TYPE:
        r"""
        Performs backpropagation for the Projection layer.
        This method overrides the `Dense.backprop` to handle specific gradient
        calculations that might arise in contexts like attention mechanisms,
        especially if inputs are 3D
        (e.g., batch_size, sequence_length, feature_dim).

        The core calculations for \( dX, dW, dB \) are similar to a standard
        Dense layer, but aggregation for \(dW\) and \(dB\) might differ.
        Given \( dZ = \frac{\partial L}{\partial Z} \) (gradient of loss w.r.t.
        layer's linear output):
        1. Gradient for weights \(W\):
           .. math::
               dW = \sum_{batch} (X^T dZ)
           Or if input X is (batch, seq_len, in_feat) and dZ is
           (batch, seq_len, out_feat),
           then \( dW = \sum_{batch} \sum_{seq} (X_{seq}^T dZ_{seq}) \).
           The current implementation sums dW across the batch dimension
           (axis 0) if the input (and thus dZ) was 3D.
        2. Gradient for bias \(b\):
           .. math::
               dB = \sum_{batch} \sum_{seq} dZ
           Summed over batch (axis 0) and sequence (axis 1 if applicable).
        3. Gradient for input \(X\):
           .. math::
               dX = dZ W^T

        Args:
            dZ (ARRAY_TYPE): Gradient of the loss w.r.t the output of this
                layer (before activation, but since activation is identity for
                typical projections, \(dZ = dA\)). Shape is expected to be
                (batch_size, sequence_length, out_features) or
                (batch_size, out_features).
            optimizer (Optimizer): Optimizer used to update weights and bias.

        Returns:
            ARRAY_TYPE: Gradient of the loss w.r.t the input of this layer
                (\(dX\)).
        """
        if self._inputs is None:
            raise ValueError("Forward pass must be called before backprop.")

        backend, _ = get_backend()

        # dZ is dL/d(output of this projection layer)
        # dZ is (batch, seq_len, out_features)
        # dW = X^T @ dZ
        # inputs_T shape: (batch, in_features, seq_len)
        # dW shape: (batch, in_features, out_features)
        dW_batched = backend.matmul(self._inputs.transpose(-1, -2), dZ)

        # Summing over the batch dimension for the final dW
        # This assumes dW_batched contains gradients for each item in the batch
        # separately, and we need a single dW for the layer's weights.
        dW = backend.sum(dW_batched, axis=0)

        # dX = dZ @ W^T
        # dX shape: (batch, seq_len, in_features)
        dX = backend.matmul(
            dZ, self._weights.T
        )  # W.T is (out_features, in_features)

        # Update weights
        dw_update, self._dw_history = optimizer.optimize(self._dw_history, dW)
        self._weights -= dw_update

        if self.add_bias and self._bias is not None:
            # dB = sum over batch and sum over sequence length of dZ
            # dZ shape: (batch, seq_len, out_features)
            # dB target shape: (1, 1, out_features)
            dB = backend.sum(dZ, axis=0, keepdims=True)
            dB = backend.sum(dB, axis=1, keepdims=True)

            db_update, self._db_history = optimizer.optimize(
                self._db_history, dB
            )
            self._bias -= db_update
        else:
            dB = None  # For retain_grad

        if self._retain_grad:
            self._dW = dW
            self._dB = dB
            self._dZ = dZ
            # This dZ is the input to backprop, i.e., dL/d(projection_output)

        return dX
