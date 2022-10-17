 
import numpy as np


class Loss:
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def backprop(self) -> np.ndarray:
        raise NotImplementedError()

class MSELoss(Loss):
    def __init__(self) -> None:
        self._y_true = None
        self._y_pred = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        self._y_true = y_true
        self._y_pred = y_pred
        return np.mean(np.square(y_pred - y_true))
    
    def backprop(self) -> np.ndarray:
        return (2 / self._y_pred.shape[0]) * (self._y_pred - self._y_true)