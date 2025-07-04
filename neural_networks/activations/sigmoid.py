from neural_networks.activations.base import Activation
from neural_networks.backend import ARRAY_TYPE, get_backend


class Sigmoid(Activation):
    """
    Sigmoid activation function.
    """

    def forward(self, inputs: ARRAY_TYPE) -> ARRAY_TYPE:
        r"""
        Performs the forward pass of the sigmoid activation function.

        The sigmoid function is defined as:
        .. math::
            \sigma(x) = \frac{1}{1 + e^{-x}}

        Args:
            inputs (ARRAY_TYPE): The input data.

        Returns:
            ARRAY_TYPE: The output of the sigmoid activation function.
        """
        super().forward(inputs)  # Stores inputs
        backend, _ = get_backend()
        self._activation = 1 / (1 + backend.exp(-1 * self._input))
        return self._activation

    def derivative(self) -> ARRAY_TYPE:
        r"""
        Computes the derivative of the sigmoid activation function.

        The derivative is calculated using the formula:
        .. math::
            \sigma'(x) = \sigma(x) (1 - \sigma(x))

        where \(\sigma(x)\) is the activation value from the forward pass.

        Returns:
            ARRAY_TYPE: The derivative of the sigmoid function with respect to
                its input, using the stored activation from the forward pass.
        """
        if (
            self._activation is None
        ):  # self._activation is set in forward pass of base class
            raise ValueError("Forward pass must be called before derivative.")
        return self._activation * (1 - self._activation)
