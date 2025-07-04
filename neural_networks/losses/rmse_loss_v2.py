from typing import Optional

import numpy as np
import torch as pt

from neural_networks.backend import ARRAY_TYPE, NUMERIC_TYPE, get_backend
from neural_networks.core.loss_base import Loss


class RMSELossV2(Loss):
    def __init__(self) -> None:
        """
        Initializes the Root Mean Squared Error (RMSE) Loss function (Version 2).

        This version calculates RMSE and its gradient directly without inheriting from MSELoss.

        Attributes:
            _y_true (Optional[ARRAY_TYPE]): Ground truth values.
            _y_pred (Optional[ARRAY_TYPE]): Predicted values.
            _loss (Optional[NUMERIC_TYPE]): The computed RMSE loss value.
            _size (Optional[int]): Total number of elements.
        """
        super().__init__()
        self._y_true: Optional[ARRAY_TYPE] = None
        self._y_pred: Optional[ARRAY_TYPE] = None
        self._loss: Optional[NUMERIC_TYPE] = None # Loss is a numeric type
        self._size: Optional[int] = None

    def forward(self, y_pred: ARRAY_TYPE, y_true: ARRAY_TYPE) -> NUMERIC_TYPE:
        r"""
        Computes the Root Mean Squared Error (RMSE) loss.

        The RMSE is calculated directly as:
        .. math::
            L_{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_{pred_i} - y_{true_i})^2}

        where:
         - \(N\) is the total number of elements in \(y_{true}\) (i.e., `_size`).
         - \(y_{pred_i}\) is the i-th predicted value.
         - \(y_{true_i}\) is the i-th true value.

        Args:
            y_pred (ARRAY_TYPE): Predicted values from the model.
            y_true (ARRAY_TYPE): True target values.

        Returns:
            NUMERIC_TYPE: The computed root mean squared error loss.
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
        self._loss = backend.sqrt(backend.mean((y_pred - y_true) ** 2))
        assert self._loss is not None
        return self._loss

    def backprop(self) -> ARRAY_TYPE:
        r"""
        Computes the gradient of the RMSE loss with respect to the predictions (\( y_{pred} \)).

        The gradient \( \frac{\partial L_{RMSE}}{\partial y_{pred}} \) is:
        .. math::
            \frac{\partial L_{RMSE}}{\partial y_{pred}} = \frac{1}{N \cdot L_{RMSE}} (y_{pred} - y_{true})

        where:
         - \(N\) is the total number of elements (`_size`).
         - \(L_{RMSE}\) is the computed RMSE loss (`_loss`).
         - \(y_{pred}\) are the predicted values.
         - \(y_{true}\) are the true values.

        A small epsilon (\(\epsilon\)) is added to \(L_{RMSE}\) in the denominator
        for numerical stability, preventing division by zero.

        Returns:
            ARRAY_TYPE: Gradient of the RMSE loss with respect to \( y_{pred} \).
        """
        assert self._y_pred is not None, "y_pred not found. Forward pass must be called before backprop."
        assert self._y_true is not None, "y_true not found. Forward pass must be called before backprop."
        assert self._loss is not None, "RMSE loss (_loss) not computed. Forward pass must be called."
        assert self._size is not None, "_size not set. Forward pass must be called."

        epsilon = 1e-16  # Small epsilon for numerical stability

        # Gradient calculation
        # (y_pred - y_true) / (N * (RMSE + epsilon))
        return (self._y_pred - self._y_true) / (
            self._size * (self._loss + epsilon)
        )
