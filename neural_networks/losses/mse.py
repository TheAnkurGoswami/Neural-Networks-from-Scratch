from typing import Optional

import numpy as np
import torch as pt

from neural_networks.backend import ARRAY_TYPE, NUMERIC_TYPE, get_backend
from neural_networks.losses.base import Loss


class MSELoss(Loss):
    def __init__(self) -> None:
        """
        Initializes the Mean Squared Error (MSE) Loss function.

        Attributes:
            _y_true (Optional[ARRAY_TYPE]): Ground truth values, stored during
                forward pass.
            _y_pred (Optional[ARRAY_TYPE]): Predicted values, stored during
                forward pass.
            _size (Optional[int]): Total number of elements, used for averaging
                the loss.
        """
        super().__init__()
        self._y_true: Optional[ARRAY_TYPE] = None
        self._y_pred: Optional[ARRAY_TYPE] = None
        self._size: Optional[int] = None

    def forward(self, y_pred: ARRAY_TYPE, y_true: ARRAY_TYPE) -> NUMERIC_TYPE:
        r"""
        Computes the Mean Squared Error (MSE) loss.

        The MSE is calculated as:
            (1/n) * Σ(y_pred - y_true)^2
        .. math::
            L_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_{pred_i} - y_{true_i})^2

        where:
         - \(N\) is the total number of elements in \(y_{true}\)
            (i.e., `_size`).
         - \(y_{pred_i}\) is the i-th predicted value.
         - \(y_{true_i}\) is the i-th true value.

        Args:
            y_pred (ARRAY_TYPE): Predicted values from the model.
            y_true (ARRAY_TYPE): True target values.

        Returns:
            NUMERIC_TYPE: The computed mean squared error loss.
        """
        backend, backend_module = get_backend()
        if backend_module == "pt":
            if isinstance(y_true, pt.Tensor):
                y_true = y_true.float()
            else:
                y_true = pt.tensor(y_true, dtype=pt.float32)
            self._size = y_true.numel()
        elif backend_module == "np":
            y_true = np.array(y_true, dtype=np.float32)
            self._size = y_true.size
        self._y_true = y_true
        self._y_pred = y_pred
        return backend.mean((y_pred - y_true) ** 2)

    def backprop(self) -> ARRAY_TYPE:
        r"""
        Computes the gradient of the MSE loss w.r.t the predictions.

        The gradient \( \frac{\partial L_{MSE}}{\partial y_{pred}} \) is
        computed as:
            ∂L/∂y_pred = (2 / n) * (y_pred - y_true)
        .. math::
            \frac{\partial L_{MSE}}{\partial y_{pred}} =
                \frac{2}{N} (y_{pred} - y_{true})

        where:
         - \(N\) is the total number of elements in \(y_{true}\)
            (i.e., `_size`).
         - \(y_{pred}\) are the predicted values.
         - \(y_{true}\) are the true values.

        Returns:
            ARRAY_TYPE: Gradient of the MSE loss w.r.t \( y_{pred} \).
        """
        assert (
            self._y_pred is not None
        ), "y_pred not found. Forward pass must be called before backprop."
        assert (
            self._y_true is not None
        ), "y_true not found. Forward pass must be called before backprop."
        assert self._size is not None
        return (2 / self._size) * (self._y_pred - self._y_true)
        # NOTE:
        """
        Why _size & not use the batch dimension i.e., _y_pred.shape[0]?
        In regression tasks, we often have a single output per sample, so we
        could use the batch dimension to compute the loss.
        However, in cases where we have multiple outputs per sample (e.g., in
        multi-output regression), using the total number of elements (_size)
        ensures that the loss is averaged correctly across all outputs.
        """
