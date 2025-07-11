from typing import Optional

import numpy as np
import torch as pt

from neural_networks.backend import ARRAY_TYPE, get_backend


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
        backend, backend_module = get_backend()
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
        - dA (ARRAY_TYPE): Gradient of the loss w.r.t the activation.

        Returns:
            - ARRAY_TYPE: Gradient of the loss w.r.t the inputs.
        """
        assert self._inputs is not None
        _, backend_module = get_backend()
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
