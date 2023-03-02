
from typing import Optional

import numpy as np


class Loss:
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backprop(self) -> np.ndarray:
        raise NotImplementedError()


class MSELoss(Loss):
    def __init__(self) -> None:
        self._y_true: Optional[np.ndarray] = None
        self._y_pred: Optional[np.ndarray] = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        self._y_true = y_true
        self._y_pred = y_pred
        return np.mean(np.square(y_pred - y_true), keepdims=True)

    def backprop(self) -> np.ndarray:
        assert self._y_pred is not None
        assert self._y_true is not None
        return (2 / self._y_pred.shape[0]) * (self._y_pred - self._y_true)


class RMSELoss(Loss):
    def __init__(self) -> None:
        self._y_true: Optional[np.ndarray] = None
        self._y_pred: Optional[np.ndarray] = None
        self._loss: Optional[np.ndarray] = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        self._y_true = y_true
        self._y_pred = y_pred
        self._loss = np.sqrt(
            np.mean(np.square(y_pred - y_true), keepdims=True))
        assert self._loss is not None
        return self._loss

    def backprop(self) -> np.ndarray:
        assert self._y_pred is not None
        assert self._y_true is not None
        assert self._loss is not None
        return np.divide(
            self._y_pred - self._y_true, self._y_pred.shape[0] * self._loss)
