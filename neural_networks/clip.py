from typing import Union

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

    def forward(self, inputs: ARRAY_TYPE) -> ARRAY_TYPE:
        self._inputs = inputs
        if backend_module == "pt":
            return backend.clamp(
                inputs, 1e-07, 1.0 - 1e-07
            )  # Avoid log(0) issues
        elif backend_module == "np":
            return np.clip(inputs, 1e-07, 1.0 - 1e-07)

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
            dZ = pt.where(
                (self._inputs > self._min_val)
                & (self._inputs < self._max_val),
                dA,
                pt.zeros_like(self._inputs),
            )
        elif backend_module == "np":
            dZ = np.where(
                (self._inputs > self._min_val)
                & (self._inputs < self._max_val),
                dA,
                0,
            )
        return dZ
