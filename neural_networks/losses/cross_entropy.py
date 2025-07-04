from typing import Optional

import numpy as np
import torch as pt

from neural_networks.backend import ARRAY_TYPE, NUMERIC_TYPE, get_backend
from neural_networks.losses.base import Loss


class CrossEntropyLoss(Loss):
    """
    CrossEntropyLoss computes the cross-entropy loss between predicted
    probabilities and true labels. This loss is commonly used in classification
    tasks.
    """

    def __init__(self) -> None:
        """
        Initializes the Cross-Entropy Loss function.

        This loss is commonly used for multi-class classification tasks where
        outputs are probabilities (e.g., after a Softmax activation).

        Attributes:
            _y_true (Optional[ARRAY_TYPE]): Ground truth labels
                (one-hot encoded).
            _y_pred (Optional[ARRAY_TYPE]): Predicted probabilities.
            _loss (Optional[NUMERIC_TYPE]): Computed cross-entropy loss.
        """
        super().__init__()
        self._y_true: Optional[ARRAY_TYPE] = None
        self._y_pred: Optional[ARRAY_TYPE] = None
        self._loss: Optional[NUMERIC_TYPE] = None

    def forward(self, y_pred: ARRAY_TYPE, y_true: ARRAY_TYPE) -> NUMERIC_TYPE:
        r"""
        Computes the forward pass of the Cross-Entropy loss function.

        For a single sample, the cross-entropy loss is:
        \( L_i = - \sum_{c=1}^{C} y_{true_{ic}} \log(y_{pred_{ic}}) \)
        where \( C \) is the number of classes.

        The total loss is the average over all \( N \) samples in the batch:
        .. math::
            L_{CE} = - \frac{1}{N} \sum_{i=1}^{N}
                \sum_{c=1}^{C} y_{true_{ic}} \log(y_{pred_{ic}})

        Args:
            y_pred (ARRAY_TYPE): Predicted probabilities from the model,
                expected to be a 2D array of shape (batch_size, num_classes).
                Values should be in the range (0, 1).
            y_true (ARRAY_TYPE): True labels, typically one-hot encoded, of
                shape (batch_size, num_classes).

        Returns:
            NUMERIC_TYPE: The computed average cross-entropy loss.

        Notes:
            - Predicted probabilities \( y_{pred} \) are clipped to a small
                range \( [\epsilon, 1-\epsilon] \) to prevent \( \log(0) \)
                which is undefined.
        """
        self._y_pred = y_pred
        backend, backend_module = get_backend()
        if backend_module == "pt":
            self._y_true = pt.tensor(y_true, dtype=pt.float32)
        elif backend_module == "np":
            self._y_true = np.array(y_true, dtype=np.float32)
        # Compute loss per sample
        sample_loss = backend.sum(
            self._y_true * backend.log(self._y_pred), dim=1
        )
        self._loss = -backend.mean(sample_loss)  # Average over all samples
        assert self._loss is not None
        return self._loss

    def backprop(self) -> ARRAY_TYPE:
        """
        Performs the backpropagation step for the loss function.
        This method computes the gradient of the loss with respect to the
        predicted values (`_y_pred`).

        The gradient is computed as:
        ∂L/∂y_pred = -y_true / y_pred

        Returns:
            ARRAY_TYPE: The gradient of the loss with respect to `_y_pred`.
        """
        assert (
            self._y_pred is not None
        ), "y_pred not found. Forward pass must be called before backprop."
        assert (
            self._y_true is not None
        ), "y_true not found. Forward pass must be called before backprop."
        assert (
            self._loss is not None
        ), "Loss not computed. Forward pass must be called."

        return (-1 / self._y_true.shape[0]) * (self._y_true / self._y_pred)
