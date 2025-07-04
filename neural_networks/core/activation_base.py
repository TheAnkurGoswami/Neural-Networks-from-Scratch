import numpy as np
import torch as pt

from neural_networks.backend import ARRAY_TYPE, get_backend


class Activation:
    """
    Base class for all activation functions.

    Activation functions introduce non-linearity into neural networks,
    allowing them to learn complex patterns.
    """

    def __init__(self) -> None:
        """
        Initializes the activation function.
        Stores the input and output of the activation function during the forward pass,
        which are often needed for the backward pass.
        """
        _, backend_module = get_backend()
        # Initialize with empty arrays of a type that matches ARRAY_TYPE philosophy
        # These will be overwritten in the forward pass.
        self._input: Optional[ARRAY_TYPE] = None
        self._activation: Optional[ARRAY_TYPE] = None


    def forward(self, inputs: ARRAY_TYPE) -> ARRAY_TYPE:
        r"""
        Performs the forward pass of the activation function.

        \( A = f(Z) \)
        where \( Z \) are the inputs (typically the linear output of a layer)
        and \( A \) is the activated output.

        This method should be implemented by subclasses. The base implementation
        stores the input and initializes a placeholder for the activation output.

        Args:
            inputs (ARRAY_TYPE): Input data (e.g., output from a linear layer).

        Returns:
            ARRAY_TYPE: Output after applying the activation function.
                        The actual computation is done in subclasses.
        """
        self._input = inputs
        # Subclasses will compute and store the actual activation in self._activation
        # and then return it.
        # For the base class, we can return a placeholder or raise NotImplementedError
        # if we expect all subclasses to fully override.
        # For now, let's assume subclasses will set and return self._activation.
        # This line is more of a placeholder to indicate self._activation will be set.
        self._activation = get_backend()[0].zeros_like(inputs)
        # raise NotImplementedError("Subclasses must implement the forward pass.")
        return self._activation # Subclass should return its computed activation

    def derivative(self) -> ARRAY_TYPE:
        r"""
        Computes the derivative of the activation function with respect to its input.

        \( f'(Z) = \frac{\partial A}{\partial Z} \)

        This method must be implemented by subclasses. It typically uses `self._input`
        or `self._activation` (the output of the forward pass) to compute the derivative.

        Returns:
            ARRAY_TYPE: Derivative of the activation function.
        """
        raise NotImplementedError("Subclasses must implement the derivative calculation.")

    def backprop(self, dA: ARRAY_TYPE) -> ARRAY_TYPE:
        r"""
        Performs the backward pass (backpropagation) of the activation function.

        Given \( dA = \frac{\partial L}{\partial A} \) (gradient of the loss \( L \)
        with respect to the activation output \( A \)), this method computes
        \( dZ = \frac{\partial L}{\partial Z} \) (gradient of the loss with respect
        to the activation input \( Z \)).

        Using the chain rule:
        .. math::
            \frac{\partial L}{\partial Z} = \frac{\partial L}{\partial A} \odot \frac{\partial A}{\partial Z}

        So, \( dZ = dA \odot f'(Z) \), where \( f'(Z) \) is the derivative of the
        activation function.

        Args:
            dA (ARRAY_TYPE): Gradient of the loss with respect to the output of this
                             activation function.

        Returns:
            ARRAY_TYPE: Gradient of the loss with respect to the input of this
                             activation function.
        """
        if self._input is None:
            raise ValueError("Forward pass must be called before backprop to store input.")
        return dA * self.derivative()
