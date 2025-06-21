from typing import Optional, Union

import numpy as np
import torch as pt

backend_module = "pt"
backend = np if backend_module == "np" else pt
ARRAY_TYPE = Union[np.ndarray, pt.Tensor]


class Clip:
    def __init__(self, min_val: float, max_val: float) -> None:
        """
        Initializes the Clip layer.

        Parameters:
        - min_val (float): Minimum value for clipping.
        - max_val (float): Maximum value for clipping.
        """
        self._min_val = min_val
        self._max_val = max_val
        self._inputs: Optional[ARRAY_TYPE] = None

    def forward(self, inputs: ARRAY_TYPE) -> ARRAY_TYPE:
        self._inputs = inputs
        if backend_module == "pt":
            return backend.clamp(
                inputs, 1e-07, 1.0 - 1e-07
            )  # Avoid log(0) issues
        if backend_module == "np":
            return np.clip(inputs, 1e-07, 1.0 - 1e-07)
        raise ValueError("Unsupported backend module")

    def backprop(self, dA: ARRAY_TYPE) -> ARRAY_TYPE:
        """
        Performs the backpropagation step for the Clip layer.

        Parameters:
        - dA (ARRAY_TYPE): Gradient of the loss with respect to the activation.

        Returns:
            - ARRAY_TYPE: Gradient of the loss with respect to the inputs.
        """
        assert self._inputs is not None
        if backend_module == "pt":
            assert isinstance(dA, pt.Tensor), "dA must be a PyTorch tensor"
            assert isinstance(
                self._inputs, pt.Tensor
            ), "Inputs must be a PyTorch tensor"
            dZ = pt.where(
                (self._inputs > self._min_val)
                & (self._inputs < self._max_val),
                dA,
                pt.zeros_like(self._inputs),
            )
        elif backend_module == "np":
            assert isinstance(dA, np.ndarray), "dA must be a NumPy array"
            assert isinstance(
                self._inputs, np.ndarray
            ), "Inputs must be a NumPy array"
            dZ = np.where(
                (self._inputs > self._min_val)
                & (self._inputs < self._max_val),
                dA,
                0,
            )
        return dZ
