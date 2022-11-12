from typing import Any, Dict, Optional, Tuple, Type

import numpy as np


class Optimizer:
    def __init__(self, **kwargs: Any) -> None:
        self._epoch = 0

    def _initialize_history(
            self, parameter: np.ndarray) -> Dict[str, np.ndarray]:
        raise NotImplementedError()

    def set_cur_epoch(self, epoch: int):
        self._epoch = epoch

    def optimize(
            self,
            history: Optional[Dict[str, np.ndarray]],
            derivative: np.ndarray) -> Tuple[
                np.ndarray, Dict[str, np.ndarray]]:
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self, learning_rate: float, momentum: float) -> None:
        super().__init__()
        self._learning_rate = learning_rate
        self._momentum = momentum

    def _initialize_history(self, parameter: np.ndarray) -> Dict[str, np.ndarray]:
        return {"accum_grad": np.zeros_like(parameter)}

    def optimize(
            self,
            history: Optional[Dict[str, np.ndarray]],
            derivative: np.ndarray) -> Tuple[
                np.ndarray, Dict[str, np.ndarray]]:
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
        if history is None:
            history = self._initialize_history(derivative)
        accum_grad = self._momentum * history["accum_grad"] + \
            self._learning_rate * derivative
        new_history = {"accum_grad": accum_grad}
        return accum_grad, new_history


class RMSProp(Optimizer):
    def __init__(
            self,
            learning_rate: float,
            rho: float,
            epsilon: float = 1e-07) -> None:
        super().__init__()
        self._learning_rate = learning_rate
        self._rho = rho
        self._epsilon = epsilon

    def _initialize_history(
            self, parameter: np.ndarray) -> Dict[str, np.ndarray]:
        return {"accum_sq_grad": np.zeros_like(parameter)}

    def optimize(
            self,
            history: Optional[Dict[str, np.ndarray]],
            derivative: np.ndarray) -> Tuple[
                np.ndarray, Dict[str, np.ndarray]]:
        if history is None:
            history = self._initialize_history(derivative)

        accum_sq_grad = self._rho * history["accum_sq_grad"] + \
            (1 - self._rho) * np.square(derivative)
        new_change = (self._learning_rate * derivative) / (
            np.sqrt(accum_sq_grad) + self._epsilon)
        new_history = {
            "accum_sq_grad": accum_sq_grad,
        }
        return new_change, new_history


class Adam(Optimizer):
    def __init__(
            self,
            learning_rate: float,
            beta1: float,
            beta2: float,
            epsilon: float = 1e-07) -> None:
        super().__init__()
        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

    def _initialize_history(
            self, parameter: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "first_moment_t": np.zeros_like(parameter),
            "second_moment_t": np.zeros_like(parameter),
        }

    def optimize(
            self,
            history: Optional[Dict[str, np.ndarray]],
            derivative: np.ndarray) -> Tuple[
                np.ndarray, Dict[str, np.ndarray]]:
        epoch = self._epoch
        if history is None:
            history = self._initialize_history(derivative)
        first_moment_t_prev = history["first_moment_t"]
        second_moment_t_prev = history["second_moment_t"]
        first_moment_t = self._beta1 * first_moment_t_prev + \
            (1 - self._beta1) * derivative
        second_moment_t = self._beta2 * second_moment_t_prev + \
            (1 - self._beta2) * np.square(derivative)
        corrected_first_moment_t = first_moment_t / (1 - self._beta1 ** epoch)
        corrected_second_moment_t = second_moment_t / (1 - self._beta2 ** epoch)

        new_change = (self._learning_rate * corrected_first_moment_t) / (
            np.sqrt(corrected_second_moment_t) + self._epsilon)

        new_history = {
            "first_moment_t": first_moment_t,
            "second_moment_t": second_moment_t,
        }

        return new_change, new_history


def get_optimizer(optimizer: str) -> Type[Optimizer]:
    optimizer_map = {
        "sgd": SGD,
        "rmsprop": RMSProp,
        "adam": Adam,
    }
    return optimizer_map[optimizer]
