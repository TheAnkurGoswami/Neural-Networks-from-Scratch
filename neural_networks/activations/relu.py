from neural_networks.activations.base import Activation
from neural_networks.backend import ARRAY_TYPE, get_backend


class ReLU(Activation):
    """
    Rectified Linear Unit (ReLU) activation function.
    """

    def forward(self, inputs: ARRAY_TYPE) -> ARRAY_TYPE:
        r"""
        Applies the ReLU (Rectified Linear Unit) activation function to the
        input data.

        The ReLU function is defined as:
        .. math::
            \text{ReLU}(x) = \max(0, x)

        Args:
            inputs (ARRAY_TYPE): The input data.

        Returns:
            ARRAY_TYPE: The output data after applying the ReLU activation
                function.
        """
        super().forward(inputs)  # Stores inputs
        backend, _ = get_backend()
        self._activation = backend.where(self._input > 0, self._input, 0)
        return self._activation

    def derivative(self) -> ARRAY_TYPE:
        r"""
        Computes the derivative of the ReLU activation function.

        The derivative of ReLU is:
        .. math::
            \text{ReLU}'(x) = \begin{cases}
                1 & \text{if } x > 0 \\
                0 & \text{if } x \leq 0
            \end{cases}

        Returns:
            ARRAY_TYPE: An array containing the derivative values for the input
                        stored during the forward pass.
        """
        backend, _ = get_backend()
        if self._input is None:
            raise ValueError("Forward pass must be called before derivative.")
        return backend.where(self._input > 0, 1, 0)
