from typing import Any, Type

import numpy as np


class Optimizer:
    def __init__(self, **kwargs: Any) -> None:
        pass

    def optimize(
            self, history: np.ndarray, derivative: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self, learning_rate: float, momentum: float) -> None:
        self._learning_rate = learning_rate
        self._momentum = momentum

    def optimize(
            self, history: np.ndarray, derivative: np.ndarray) -> np.ndarray:
        """
        The original implementation for stochastic gradient descent is
        as follows:
            V_{i} = momentum * V_{i-1} + (1- momentum) * dW
            W = W - learning_rate * V_{i}

        Exact formulation:
            new_history = self._momentum * history + \
                (1 - self._momentum) * derivative
            new_change = self._learning_rate * new_history

        But most implementations use an approximated & rescaled formula, which
        is given below:
            new_change = momentum * old_change + learning_rate * dW
            W = W - new_change
        """
        new_change = self._momentum * history + \
            self._learning_rate * derivative
        return new_change


def get_optimizer(optimizer: str) -> Type[Optimizer]:
    optimizer_map = {
        "sgd": SGD,
    }
    return optimizer_map[optimizer]
