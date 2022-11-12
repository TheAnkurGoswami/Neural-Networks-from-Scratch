from typing import Any, Tuple, Type

import numpy as np


class Optimizer:
    def __init__(self, **kwargs: Any) -> None:
        pass

    def optimize(
            self, history: np.ndarray,
            derivative: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self, learning_rate: float, momentum: float) -> None:
        self._learning_rate = learning_rate
        self._momentum = momentum

    def optimize(
            self, history: np.ndarray,
            derivative: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        return new_change, new_change


class RMSProp(Optimizer):
    def __init__(
            self,
            learning_rate: float,
            rho: float,
            epsilon: float = 1e-07) -> None:
        self._learning_rate = learning_rate
        self._rho = rho
        self._epsilon = epsilon

    def optimize(
            self, history: np.ndarray,
            derivative: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        new_history = self._rho * history + \
            (1 - self._rho) * np.square(derivative)
        new_change = (self._learning_rate * derivative) / (
            np.sqrt(new_history) + self._epsilon)
        return new_change, new_history


def get_optimizer(optimizer: str) -> Type[Optimizer]:
    optimizer_map = {
        "sgd": SGD,
        "rmsprop": RMSProp,
    }
    return optimizer_map[optimizer]
