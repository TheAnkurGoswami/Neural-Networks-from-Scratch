from neural_networks.activations.base import Activation
from neural_networks.backend import ARRAY_TYPE, get_backend


class Identity(Activation):
    """
    Identity activation function.
    Returns the input directly.
    """

    def forward(self, inputs: ARRAY_TYPE) -> ARRAY_TYPE:
        r"""
        Performs the forward pass for the Identity activation function.

        The output is the same as the input.
        .. math::
            \text{Identity}(x) = x

        Args:
            inputs (ARRAY_TYPE): The input data to the activation function.

        Returns:
            ARRAY_TYPE: The output after applying the activation function
                (same as input).
        """
        # Store inputs for derivative calculation if needed by base class or
        # for debugging
        super().forward(inputs)
        self._activation = inputs  # Activation output is the input itself
        return inputs

    def derivative(self) -> ARRAY_TYPE:
        r"""
        Computes the derivative of the Identity activation function.

        The derivative of the Identity function is always 1.
        .. math::
            \frac{d}{dx} \text{Identity}(x) = 1

        Returns:
            ARRAY_TYPE: An array of ones with the same shape as the input
                stored during the forward pass.
        """
        backend, _ = get_backend()
        if self._input is None:
            raise ValueError("Forward pass must be called before derivative.")
        return backend.ones_like(self._input)
