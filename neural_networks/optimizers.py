from typing import Dict, Optional, Tuple, Type

import numpy as np


class Optimizer:
    def __init__(self) -> None:
        self._epoch = 0

    def _initialize_history(self, parameter: np.ndarray) -> Dict[str, np.ndarray]:
        raise NotImplementedError()

    def set_cur_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def optimize(
        self, history: Optional[Dict[str, np.ndarray]], derivative: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        raise NotImplementedError()


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer with optional momentum.
    Parameters
    ----------
    learning_rate : float
        The learning rate for the optimizer.
    momentum : float, optional
        The momentum factor (default is 0).
    """

    def __init__(self, learning_rate: float, momentum: float = 0) -> None:
        super().__init__()
        self._learning_rate = learning_rate
        self._momentum = momentum

    def _initialize_history(self, parameter: np.ndarray) -> Dict[str, np.ndarray]:
        return {"accum_grad": np.zeros_like(parameter)}

    def optimize(
        self, history: Optional[Dict[str, np.ndarray]], derivative: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
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
        # Initialize history if it's None
        if history is None:
            history = self._initialize_history(derivative)

        # Update the accumulated gradient with the current gradient
        accum_grad = (
            self._momentum * history["accum_grad"] + self._learning_rate * derivative
        )

        # Store the updated accumulated gradient in history
        new_history = {"accum_grad": accum_grad}
        return accum_grad, new_history


class RMSProp(Optimizer):
    """
    RMSProp optimizer.
    RMSProp is an adaptive learning rate method that divides the learning rate
    for a weight by a running average of the magnitudes of recent gradients f
    or that weight. It maintains a moving (discounted) average of the square of
    gradients and divides the gradient by the root of this average.

    Attributes:
        learning_rate (float): The learning rate for the optimizer.
        rho (float): Discounting factor for the history/coming gradient.
        epsilon (float): A small constant for numerical stability (default is 1e-07).
    """

    def __init__(
        self, learning_rate: float, rho: float, epsilon: float = 1e-07
    ) -> None:
        super().__init__()
        self._learning_rate = learning_rate
        self._rho = rho
        self._epsilon = epsilon

    def _initialize_history(self, parameter: np.ndarray) -> Dict[str, np.ndarray]:
        return {"accum_sq_grad": np.zeros_like(parameter)}

    def optimize(
        self, history: Optional[Dict[str, np.ndarray]], derivative: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Performs an optimization step using the RMSprop algorithm.

        Formulation:
            accum_sq_grad = rho * accum_sq_grad + (1 - rho) * (derivative^2)
            new_change = (learning_rate * derivative) / (sqrt(accum_sq_grad) + epsilon)
        """
        # Initialize history if it's None
        if history is None:
            history = self._initialize_history(derivative)

        # Update the accumulated squared gradient with the current gradient
        accum_sq_grad = self._rho * history["accum_sq_grad"] + (
            1 - self._rho
        ) * np.square(derivative)

        # Compute the parameter update using the RMSProp formula
        new_change = (self._learning_rate * derivative) / (
            np.sqrt(accum_sq_grad) + self._epsilon
        )

        # Store the updated accumulated squared gradient in history
        new_history = {
            "accum_sq_grad": accum_sq_grad,
        }

        # Increment the epoch counter
        self._epoch += 1

        return new_change, new_history


class Adam(Optimizer):
    """
    The Adam optimizer is an adaptive learning rate optimization algorithm that's been designed specifically for training deep neural networks.
    It combines the advantages of two other extensions of stochastic gradient descent, namely AdaGrad and RMSProp.
    Attributes:
        learning_rate (float): The learning rate or step size used for updating the parameters.
        beta_1 (float): The exponential decay rate for the first moment estimates. Default is 0.9.
        beta_2 (float): The exponential decay rate for the second moment estimates. Default is 0.999.
        epsilon (float): A small constant for numerical stability. Default is 1e-07.
    """

    def __init__(
        self,
        learning_rate: float,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-07,
    ) -> None:
        super().__init__()
        self._learning_rate = learning_rate
        self._beta1 = beta_1
        self._beta2 = beta_2
        self._epsilon = epsilon

    def _initialize_history(self, parameter: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "first_moment_t": np.zeros_like(
                parameter,
            ),
            "second_moment_t": np.zeros_like(parameter),
        }

    def optimize(
        self, history: Optional[Dict[str, np.ndarray]], derivative: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:

        # Get the current epoch
        epoch = self._epoch

        # Initialize history if it's None
        if history is None:
            history = self._initialize_history(derivative)

        # Retrieve previous first and second moment estimates from history
        first_moment_t_prev = history["first_moment_t"]
        second_moment_t_prev = history["second_moment_t"]

        # Update biased first moment estimate
        first_moment_t = (
            self._beta1 * first_moment_t_prev + (1 - self._beta1) * derivative
        )

        # Update biased second raw moment estimate
        second_moment_t = self._beta2 * second_moment_t_prev + (
            1 - self._beta2
        ) * np.square(derivative)

        # Compute bias-corrected first moment estimate
        first_mom_corr = 1 - np.power(self._beta1, epoch)
        corrected_first_moment_t = first_moment_t / first_mom_corr

        # Compute bias-corrected second raw moment estimate
        second_mom_corr = 1 - np.power(self._beta2, epoch)
        corrected_second_moment_t = second_moment_t / second_mom_corr

        # Compute the denominator for the update rule
        denom = np.sqrt(corrected_second_moment_t) + self._epsilon

        # Compute the parameter update
        new_change = ((self._learning_rate) * corrected_first_moment_t) / denom

        # Store the updated first and second moment estimates in history
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
