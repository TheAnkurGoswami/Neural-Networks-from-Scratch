from typing import Optional, Union

import numpy as np
import torch as pt
import logging

backend_module = "pt"
backend = np if backend_module == "np" else pt
ARRAY_TYPE = Union[np.ndarray, pt.Tensor]


class Loss:
    def forward(self, y_pred: ARRAY_TYPE, y_true: ARRAY_TYPE) -> float:
        """
        Computes the forward pass of the loss function.

        Args:
            y_pred (ARRAY_TYPE): Predicted values.
            y_true (ARRAY_TYPE): True values.

        Returns:
            float: Computed loss.
        """
        raise NotImplementedError()

    def backprop(self) -> ARRAY_TYPE:
        """
        Computes the backward pass of the loss function.

        Returns:
            ARRAY_TYPE: Gradient of the loss with respect to the predictions.
        """
        raise NotImplementedError()


class MSELoss(Loss):
    def __init__(self) -> None:
        """
        Initializes the MSELoss class.
        """
        self._y_true: Optional[ARRAY_TYPE] = None
        self._y_pred: Optional[ARRAY_TYPE] = None

    def forward(self, y_pred: ARRAY_TYPE, y_true: ARRAY_TYPE) -> ARRAY_TYPE:
        """
        Computes the Mean Squared Error (MSE) loss.
        Formula for MSE loss: (1/n) * Σ(y_pred - y_true)^2

        Args:
            y_pred (ARRAY_TYPE): Predicted values.
            y_true (ARRAY_TYPE): True values.

        Returns:
            ARRAY_TYPE: Computed MSE loss.
        """
        self._y_true = y_true
        self._y_pred = y_pred
        return backend.mean((y_pred - y_true) ** 2)

    def backprop(self) -> ARRAY_TYPE:
        """
        Computes the gradient of the MSE loss with respect to the predictions.

        The gradient is computed as:
        ∂L/∂y_pred = (2 / n) * (y_pred - y_true)

        Returns:
            ARRAY_TYPE: Gradient of the MSE loss.
        """
        assert self._y_pred is not None
        assert self._y_true is not None
        return (2 / self._y_pred.size(0)) * (self._y_pred - self._y_true)


class RMSELoss(Loss):
    def __init__(self) -> None:
        """
        Initializes the RMSELoss class.
        """
        self._y_true: Optional[ARRAY_TYPE] = None
        self._y_pred: Optional[ARRAY_TYPE] = None
        self._loss: Optional[ARRAY_TYPE] = None

    def forward(self, y_pred: ARRAY_TYPE, y_true: ARRAY_TYPE) -> ARRAY_TYPE:
        """
        Computes the Root Mean Squared Error (RMSE) loss.
        Formula for RMSE loss: sqrt((1/n) * Σ(y_pred - y_true)^2)

        Args:
            y_pred (ARRAY_TYPE): Predicted values.
            y_true (ARRAY_TYPE): True values.

        Returns:
            ARRAY_TYPE: Computed RMSE loss.
        """
        self._y_true = y_true
        self._y_pred = y_pred
        self._loss = backend.sqrt(backend.mean((y_pred - y_true) ** 2))
        assert self._loss is not None
        return self._loss

    def backprop(self) -> ARRAY_TYPE:
        """
        Computes the gradient of the RMSE loss with respect to the predictions.

        The gradient is computed as:
        ∂L/∂y_pred = (1 / (n * RMSE)) * (y_pred - y_true)

        Returns:
            ARRAY_TYPE: Gradient of the RMSE loss.
        """
        assert self._y_pred is not None
        assert self._y_true is not None
        assert self._loss is not None
        return (self._y_pred - self._y_true) / (
            self._y_pred.size(0) * self._loss
        )


class CrossEntropyLoss(Loss):
    """
    CrossEntropyLoss computes the cross-entropy loss between predicted
    probabilities and true labels. This loss is commonly used in classification
    tasks.
    """

    def __init__(self):
        self._y_true: Optional[ARRAY_TYPE] = None
        self._y_pred: Optional[ARRAY_TYPE] = None
        self._loss: Optional[ARRAY_TYPE] = None

    def forward(self, y_pred: ARRAY_TYPE, y_true: ARRAY_TYPE) -> ARRAY_TYPE:
        """
        Computes the forward pass of the loss function.
        Formula for CrossEntropy loss: -Σ(y_true * log(y_pred))

        Args:
            y_pred (ARRAY_TYPE): Predicted probabilities, a 2D tensor where each
                row corresponds to the predicted probabilities for a single
                sample.
            y_true (ARRAY_TYPE): True labels, a 2D tensor where each row
                corresponds to the one-hot encoded true labels for a single
                sample.
        Returns:
            ARRAY_TYPE: The computed loss value.
        Notes:
            - Clips the predicted probabilities to avoid numerical instability
                when taking the logarithm.
        """
        if backend_module == "pt":
            self._y_true = pt.tensor(y_true, dtype=pt.float32)
            self._y_pred = backend.clamp(
                y_pred, 1e-07, 1.0 - 1e-07
            )  # Avoid log(0) issues
        elif backend_module == "np":
            self._y_true = np.array(y_true, dtype=np.float32)
            self._y_pred = np.clip(
                y_pred, 1e-07, 1.0 - 1e-07
            )  # Avoid log(0) issues
        # Compute loss per sample
        sample_loss = backend.sum(
            self._y_true * backend.log(self._y_pred), dim=1
        )
        self._loss = -backend.mean(sample_loss)  # Average over all samples
        logging.info(f"log(y_pred): {backend.log(self._y_pred)}")
        logging.info(f"y_pred: {self._y_pred}")
        logging.info(
            f"CrossEntropyLoss forward pass computed with loss: {self._loss.item()}"
        )
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
        assert self._y_pred is not None
        assert self._y_true is not None
        assert self._loss is not None
        return (-1 / self._y_true.size(0)) * (self._y_true / self._y_pred)
