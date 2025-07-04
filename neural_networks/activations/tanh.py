from neural_networks.backend import ARRAY_TYPE, get_backend
from neural_networks.core.activation_base import Activation


class Tanh(Activation):
    """
    Hyperbolic Tangent (Tanh) activation function.
    """

    def forward(self, inputs: ARRAY_TYPE) -> ARRAY_TYPE:
        r"""
        Performs the forward pass using the hyperbolic tangent (tanh) activation function.

        The tanh activation function is defined as:
        .. math::
            \text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}

        Args:
            inputs (ARRAY_TYPE): The input data to the activation function.

        Returns:
            ARRAY_TYPE: The output after applying the tanh activation function.
        """
        super().forward(inputs) # Stores inputs
        backend, _ = get_backend()
        self._activation = backend.tanh(self._input)
        return self._activation

    def derivative(self) -> ARRAY_TYPE:
        r"""
        Computes the derivative of the tanh activation function.

        The derivative is calculated using the formula:
        .. math::
            \text{tanh}'(x) = 1 - \text{tanh}^2(x)

        where \(\text{tanh}(x)\) is the activation value from the forward pass.

        Returns:
            ARRAY_TYPE: The derivative of the tanh function with respect to its input,
                        using the stored activation from the forward pass.
        """
        backend, _ = get_backend()
        if self._activation is None: # self._activation is set in forward pass of base class
            raise ValueError("Forward pass must be called before derivative.")
        return 1 - backend.square(self._activation)
