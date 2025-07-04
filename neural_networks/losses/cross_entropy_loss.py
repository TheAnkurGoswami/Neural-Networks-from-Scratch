from typing import Optional

import numpy as np
import torch as pt

from neural_networks.backend import ARRAY_TYPE, NUMERIC_TYPE, get_backend
from neural_networks.core.loss_base import Loss


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
            _y_true (Optional[ARRAY_TYPE]): Ground truth labels (one-hot encoded).
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
            L_{CE} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{true_{ic}} \log(y_{pred_{ic}})

        Args:
            y_pred (ARRAY_TYPE): Predicted probabilities from the model, expected to be a 2D
                                 array of shape (batch_size, num_classes).
                                 Values should be in the range (0, 1).
            y_true (ARRAY_TYPE): True labels, typically one-hot encoded, of shape
                                 (batch_size, num_classes).

        Returns:
            NUMERIC_TYPE: The computed average cross-entropy loss.

        Notes:
            - Predicted probabilities \( y_{pred} \) are clipped to a small range
              \( [\epsilon, 1-\epsilon] \) to prevent \( \log(0) \) which is undefined.
        """
        self._y_pred = y_pred
        backend, backend_module = get_backend()

        # Ensure y_true is the correct type and backend
        if backend_module == "pt":
            if not isinstance(y_true, pt.Tensor):
                self._y_true = pt.tensor(y_true, dtype=pt.float32)
            else:
                self._y_true = y_true.float() # Ensure float32 for consistency
        elif backend_module == "np":
            if not isinstance(y_true, np.ndarray):
                self._y_true = np.array(y_true, dtype=np.float32)
            else:
                self._y_true = y_true.astype(np.float32) # Ensure float32

        # Add a small epsilon to y_pred to prevent log(0)
        epsilon = 1e-12
        y_pred_clipped = backend.clip(self._y_pred, epsilon, 1.0 - epsilon)

        # Compute loss per sample
        sample_loss = backend.sum(
            self._y_true * backend.log(y_pred_clipped), dim=1
        )
        self._loss = -backend.mean(sample_loss)  # Average over all samples
        assert self._loss is not None
        return self._loss

    def backprop(self) -> ARRAY_TYPE:
        r"""
        Performs the backpropagation step for the Cross-Entropy loss function.

        This computes the gradient of the loss with respect to the predicted probabilities
        (\( \frac{\partial L_{CE}}{\partial y_{pred}} \)).

        .. math::
            \frac{\partial L_{CE}}{\partial y_{pred_{ic}}} = - \frac{1}{N} \frac{y_{true_{ic}}}{y_{pred_{ic}}}

        where:
         - \(N\) is the batch size (number of samples).
         - \(y_{true_{ic}}\) is the true label for sample \(i\), class \(c\).
         - \(y_{pred_{ic}}\) is the predicted probability for sample \(i\), class \(c\).

        Args:
            None (uses stored `_y_pred`, `_y_true` from the forward pass).

        Returns:
            ARRAY_TYPE: The gradient of the loss with respect to \( y_{pred} \).

        Notes:
            - \( y_{pred} \) is clipped to \( [\epsilon, 1-\epsilon] \) during gradient
              calculation to prevent division by zero.
        """
        assert self._y_pred is not None, "y_pred not found. Forward pass must be called before backprop."
        assert self._y_true is not None, "y_true not found. Forward pass must be called before backprop."
        assert self._loss is not None, "Loss not computed. Forward pass must be called."

        # Add a small epsilon to y_pred to prevent division by zero, consistent with forward pass.
        epsilon = 1e-12
        # Ensure y_pred is clipped for gradient calculation as well
        # Accessing backend via get_backend() to ensure it's the current one.
        backend, _ = get_backend()
        y_pred_clipped = backend.clip(self._y_pred, epsilon, 1.0 - epsilon)

        # N (batch size) is the first dimension of y_true
        if self._y_true.ndim == 0 : # Should not happen with batch_size > 0
            batch_size = 1
        elif self._y_true.shape[0] == 0: # Should not happen with data
             batch_size = 1 # Avoid division by zero, though this case is problematic
        else:
            batch_size = self._y_true.shape[0]
            if batch_size == 0: batch_size = 1 # Defensive

        return (-1.0 / batch_size) * (self._y_true / y_pred_clipped)
