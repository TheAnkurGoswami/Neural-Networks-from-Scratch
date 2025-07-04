import numpy as np
import torch as pt

from neural_networks.activations.base import Activation
from neural_networks.backend import ARRAY_TYPE, get_backend
from neural_networks.clip import Clip


class Softmax(Activation):
    """
    Softmax activation function.
    Normalizes input into a probability distribution.
    """

    def __init__(self, dim: int = 1, do_clip: bool = True) -> None:
        r"""
        Initializes the Softmax activation function.

        Args:
            dim (int, optional): The dimension along which Softmax will be
                computed. Defaults to 1 (typically the feature dimension).
            do_clip (bool, optional): Whether to clip the output probabilities
                to avoid numerical instability (e.g., log(0)).
                Defaults to True.
        """
        super().__init__()
        self.dim = dim
        self.do_clip = do_clip
        if do_clip:
            # Epsilon values for clipping to prevent log(0) or log(1) issues
            # downstream
            self.clip_fn = Clip(min_val=1e-07,max_val=1.0 - 1e-07)

    def forward(self, inputs: ARRAY_TYPE) -> ARRAY_TYPE:
        r"""
        Performs the forward pass of the Softmax activation function.
        This method computes the activation values using the softmax function:
            softmax(x_i) = exp(x_i) / Σ(exp(x_j))
        where x_i is an input value, and the denominator is the sum of the
        exponentials of all input values.

        The Softmax function is defined as:
        .. math::
            \text{Softmax}(x_i) =
                \frac{e^{x_i - \max(x)}}{\sum_{j} e^{x_j - \max(x)}}

        Subtracting max(x) from x improves numerical stability without
        changing the output.

        Args:
            inputs (ARRAY_TYPE): The input array (logits).

        Returns:
            ARRAY_TYPE: The computed activation values (probabilities).
        """
        super().forward(inputs)  # Stores original inputs as self._input
        backend, backend_module = get_backend()

        # Numerical stability: subtract max for each sample along the
        # specified dimension
        if backend_module == "np":
            assert isinstance(
                self._input, np.ndarray
            ), "Inputs must be a NumPy array for np backend"
            # Ensure keepdims=True to allow broadcasting
            stable_inputs = self._input - np.max(
                self._input, axis=self.dim, keepdims=True
            )
        elif backend_module == "pt":
            assert isinstance(
                self._input, pt.Tensor
            ), "Inputs must be a PyTorch tensor for pt backend"
            # Ensure keepdims=True to allow broadcasting
            stable_inputs = (
                self._input
                - pt.max(self._input, dim=self.dim, keepdim=True).values
            )
        else:
            # Fallback or error for unsupported backends if any
            raise NotImplementedError(
                f"Backend {backend_module} not supported for Softmax stability"
                "adjustment."
            )

        exp_values = backend.exp(stable_inputs)
        sum_exp_values = backend.sum(exp_values, dim=self.dim, keepdim=True)

        self._activation = exp_values / sum_exp_values

        if self.do_clip:
            self._activation = self.clip_fn.forward(self._activation)

        return self._activation

    def backprop(self, dA: ARRAY_TYPE) -> ARRAY_TYPE:
        r"""
        Performs the backward pass for the Softmax activation function.

        Given the gradient of the loss w.r.t the Softmax output
        (\( \frac{\partial L}{\partial a} = dA \)),
        this computes the gradient of the loss w.r.t the Softmax input
        (\( \frac{\partial L}{\partial z} \)).

        The calculation is:
        .. math::
            \frac{\partial L}{\partial z_k} =
            \sum_{i}
            \frac{\partial L}{\partial a_i} \frac{\partial a_i}{\partial z_k}

        For Softmax, a common simplification for the combined derivative
        \( \frac{\partial L}{\partial z} \) when used with Cross-Entropy loss
        is often computed directly in the loss function's backprop.

        However, if computing the Jacobian explicitly:
        \( J_{ik} = a_i (\delta_{ik} - a_k) \),
        where \( a_i \) is \( \text{Softmax}(z_i) \).
        Then \( \frac{\partial L}{\partial z_k} =
            \sum_i dA_i \cdot a_i(\delta_{ik} - a_k) \)
        This can be simplified to:
        .. math::
            dZ_k = a_k \left( dA_k - \sum_j dA_j \cdot a_j \right)

        So, \( dZ = \text{act} \odot
                (dA - \text{sum}(
                    dA \odot \text{act}, \text{dim}, \text{keepdim=True})) \)
        where \( \odot \) is element-wise multiplication.

        Args:
            dA (ARRAY_TYPE): Gradient of the loss w.r.t the activation output.

        Returns:
            ARRAY_TYPE: Gradient of the loss w.r.t the activation input
                (logits).
        """
        if self._activation is None:
            raise ValueError("Forward pass must be called before backprop.")

        # If clipping was applied in forward, apply its backprop first
        if self.do_clip:
            dA = self.clip_fn.backprop(dA)

        # Element-wise product of dA and activation
        # For each sample in the batch, computes sum(dA_i * activation_i)
        # dA_times_act = dA * self._activation # This is done inside sum for
        # some backends

        dot_product = get_backend()[0].sum(
            dA * self._activation, dim=self.dim, keepdim=True
        )

        # dZ = activation * (dA - sum_dA_act)
        dZ = self._activation * (dA - dot_product)

        return dZ

    def derivative(self) -> ARRAY_TYPE:
        r"""
        Computes the Jacobian matrix of the Softmax function for each sample in
        the batch.

        The Jacobian matrix \( J \) for a single sample's Softmax output
        \( a \) w.r.t its input \( z \) has elements
            \( J_{ij} = \frac{\partial a_i}{\partial z_j} \).

        .. math::
            J_{ij} = \begin{cases}
                        a_i (1 - a_i) & \text{if } i = j \\
                        -a_i a_j & \text{if } i \neq j
                    \end{cases}


        Returns:
            ARRAY_TYPE: A tensor where each slice [b, :, :] is the Jacobian
                matrix for the b-th sample.
                Shape: (batch_size, num_classes, num_classes).
        """
        if self._activation is None:
            raise ValueError("Forward pass must be called before derivative.")

        batch_size, num_classes = self._activation.shape
        backend, _ = get_backend()
        jacobian_mat = backend.zeros(batch_size, num_classes, num_classes)
        for batch_idx in range(batch_size):
            for row_idx in range(num_classes):
                for col_idx in range(row_idx, num_classes):
                    if row_idx == col_idx:
                        jacobian_mat[batch_idx, row_idx, col_idx] = (
                            self._activation[batch_idx, row_idx]
                            * (1 - self._activation[batch_idx, row_idx])
                        )
                    else:
                        jacobian_mat[batch_idx, row_idx, col_idx] = (
                            jacobian_mat[batch_idx, col_idx, row_idx]
                        ) = (
                            -self._activation[batch_idx, row_idx]
                            * self._activation[batch_idx, col_idx]
                        )
        return jacobian_mat

    # backprop_v2 using the Jacobian might be less efficient than the direct
    # backprop method above but is included for completeness or specific use
    # cases.
    def backprop_v2(self, dA: ARRAY_TYPE) -> ARRAY_TYPE:
        """
        Alternative backpropagation using the explicitly computed Jacobian.
        dZ = dA @ J (for each sample in batch)
        """
        if self._activation is None:
            raise ValueError("Forward pass must be called before backprop_v2.")

        backend, _ = get_backend()
        jac_mat = self.derivative()  # (batch_size, num_classes, num_classes)
        dZ_arr = []
        if self.do_clip:
            dA = self.clip.backprop(dA)
        for batch_idx in range(jac_mat.shape[0]):
            dZ = backend.matmul(
                dA[batch_idx : batch_idx + 1, :], jac_mat[batch_idx]
            )
            dZ_arr.append(dZ.flatten())
        return backend.stack(dZ_arr)
