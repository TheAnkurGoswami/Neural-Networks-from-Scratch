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

        Args:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): True values.

        Returns:
            float: Computed RMSE loss.
        """
        self._y_true = y_true
        self._y_pred = y_pred
        self._loss = np.sqrt(
            np.mean(np.square(y_pred - y_true), keepdims=True))
        assert self._loss is not None
        return self._loss

    def backprop(self) -> np.ndarray:
        """
        Computes the gradient of the RMSE loss with respect to the predictions.

        Returns:
            np.ndarray: Gradient of the RMSE loss.
        """
        assert self._y_pred is not None
        assert self._y_true is not None
        assert self._loss is not None
        return np.divide(
            self._y_pred - self._y_true, self._y_pred.shape[0] * self._loss
        )
