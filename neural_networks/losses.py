from typing import Optional

import numpy as np


class Loss:
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes the forward pass of the loss function.

        Args:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): True values.

        Returns:
            float: Computed loss.
        """
        raise NotImplementedError()

    def backprop(self) -> np.ndarray:
        """
        Computes the backward pass of the loss function.

        Returns:
            np.ndarray: Gradient of the loss with respect to the predictions.
        """
        raise NotImplementedError()


class MSELoss(Loss):
    def __init__(self) -> None:
        """
        Initializes the MSELoss class.
        """
        self._y_true: Optional[np.ndarray] = None
        self._y_pred: Optional[np.ndarray] = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes the Mean Squared Error (MSE) loss.
        Formula for MSE loss: (1/n) * Σ(y_pred - y_true)^2

        Args:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): True values.

        Returns:
            float: Computed MSE loss.
        """
        self._y_true = y_true
        self._y_pred = y_pred
        return np.mean(np.square(y_pred - y_true), keepdims=True)

    def backprop(self) -> np.ndarray:
        """
        Computes the gradient of the MSE loss with respect to the predictions.

        The gradient is computed as:
        ∂L/∂y_pred = (2 / n) * (y_pred - y_true)

        where:
        - L is the loss
        - y_pred is the predicted values
        - y_true is the true values
        - n is the number of samples

        Returns:
            np.ndarray: Gradient of the MSE loss.
        """
        assert self._y_pred is not None
        assert self._y_true is not None
        return (2 / self._y_pred.shape[0]) * (self._y_pred - self._y_true)


class RMSELoss(Loss):
    def __init__(self) -> None:
        """
        Initializes the RMSELoss class.
        """
        self._y_true: Optional[np.ndarray] = None
        self._y_pred: Optional[np.ndarray] = None
        self._loss: Optional[np.ndarray] = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes the Root Mean Squared Error (RMSE) loss.
        Formula for RMSE loss: sqrt((1/n) * Σ(y_pred - y_true)^2)

        Args:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): True values.

        Returns:
            float: Computed RMSE loss.
        """
        self._y_true = y_true
        self._y_pred = y_pred
        self._loss = np.sqrt(
            np.mean(np.square(y_pred - y_true), keepdims=True)
        )
        assert self._loss is not None
        return self._loss

    def backprop(self) -> np.ndarray:
        """
        Computes the gradient of the RMSE loss with respect to the predictions.

        The gradient is computed as:
        ∂L/∂y_pred = (1 / (n * RMSE)) * (y_pred - y_true)

        where:
        - L is the loss
        - y_pred is the predicted values
        - y_true is the true values
        - n is the number of samples
        - RMSE is the root mean squared error

        Returns:
            np.ndarray: Gradient of the RMSE loss.
        """
        assert self._y_pred is not None
        assert self._y_true is not None
        assert self._loss is not None
        return np.divide(
            self._y_pred - self._y_true, self._y_pred.shape[0] * self._loss
        )


class CrossEntropyLoss(Loss):
    """
    CrossEntropyLoss computes the cross-entropy loss between predicted
    probabilities and true labels. This loss is commonly used in classification
    tasks.
    """
    def __init__(self):
        self._y_true: Optional[np.ndarray] = None
        self._y_pred: Optional[np.ndarray] = None
        self._loss: Optional[np.ndarray] = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes the forward pass of the loss function.
        Formula for CrossEntropy loss: -Σ(y_true * log(y_pred))

        Args:
            y_pred (np.ndarray): Predicted probabilities, a 2D array where each
                row corresponds to the predicted probabilities for a single
                sample.
            y_true (np.ndarray): True labels, a 2D array where each row
                corresponds to the one-hot encoded true labels for a single
                sample.
        Returns:
            float: The computed loss value.
        Notes:
            - Clips the predicted probabilities to avoid numerical instability
                when taking the logarithm.
            - The loss is calculated as the negative log likelihood for the
                given true labels and predicted probabilities.
        """

        self._y_true = y_true
        self._y_pred = y_pred
        self._y_pred = np.clip(y_pred, 1e-07, 1.0 - 1e-07)
        self._loss = -1 * np.sum(y_true * np.log(self._y_pred))
        assert self._loss is not None
        return self._loss

    def backprop(self) -> np.ndarray:
        """
        Performs the backpropagation step for the loss function.
        This method computes the gradient of the loss with respect to the
        predicted values (`_y_pred`).

        The gradient is computed as:
        ∂L/∂y_pred = -y_true / y_pred

        where:
            - L is the loss
            - y_pred is the predicted values
            - y_true is the true values

        Returns:
            np.ndarray: The gradient of the loss with respect to `_y_pred`.
        Raises:
            AssertionError: If `_y_pred`, `_y_true`, or `_loss` is not set.
        """

        assert self._y_pred is not None
        assert self._y_true is not None
        assert self._loss is not None
        return -1 * np.divide(self._y_true, self._y_pred)
