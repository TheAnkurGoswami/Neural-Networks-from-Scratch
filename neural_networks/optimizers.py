from typing import Dict, Optional, Tuple, Type

from neural_networks.backend import ARRAY_TYPE, get_backend


class Optimizer:
    """
    Base class for all optimizers.
    """

    def __init__(self) -> None:
        self._epoch = 0

    def _initialize_history(
        self, parameter: ARRAY_TYPE
    ) -> Dict[str, ARRAY_TYPE]:
        """
        Initializes the history for the optimizer.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError()

    def set_cur_epoch(self, epoch: int) -> None:
        """
        Sets the current epoch for the optimizer.
        """
        self._epoch = epoch

    def optimize(
        self, history: Optional[Dict[str, ARRAY_TYPE]], derivative: ARRAY_TYPE
    ) -> Tuple[ARRAY_TYPE, Dict[str, ARRAY_TYPE]]:
        """
        Performs an optimization step.
        This method should be overridden by subclasses.
        """
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

    def _initialize_history(
        self, parameter: ARRAY_TYPE
    ) -> Dict[str, ARRAY_TYPE]:
        """
        Initializes the history for the SGD optimizer.
        """
        backend, _ = get_backend()
        return {"accum_grad": backend.zeros_like(parameter)}

    def optimize(
        self, history: Optional[Dict[str, ARRAY_TYPE]], derivative: ARRAY_TYPE
    ) -> Tuple[ARRAY_TYPE, Dict[str, ARRAY_TYPE]]:
        """
        Performs an optimization step using the SGD algorithm.

        Parameters
        ----------
        history : Optional[Dict[str, ARRAY_TYPE]]
            The history of the optimizer.
        derivative : ARRAY_TYPE
            The gradient of the loss function with respect to the parameters.

        Returns
        -------
        Tuple[ARRAY_TYPE, Dict[str, ARRAY_TYPE]]
            The parameter update and the updated history.
        """
        # Initialize history if it's None
        if history is None:
            history = self._initialize_history(derivative)

        # Update the accumulated gradient with the current gradient
        accum_grad = self._momentum * history["accum_grad"] + derivative

        # Store the updated accumulated gradient in history
        new_history = {"accum_grad": accum_grad}
        return self._learning_rate * accum_grad, new_history


class RMSProp(Optimizer):
    """
    RMSProp optimizer.
    RMSProp is an adaptive learning rate method that divides the learning rate
    for a weight by a running average of the magnitudes of recent gradients
    for that weight. It maintains a moving (discounted) average of the square
    of gradients and divides the gradient by the root of this average.

    Parameters
    ----------
    learning_rate : float
        The learning rate for the optimizer.
    rho : float
        Discounting factor for the history/coming gradient.
    epsilon : float, optional
        A small constant for numerical stability (default is 1e-07).
    """

    def __init__(
        self, learning_rate: float, rho: float, epsilon: float = 1e-07
    ) -> None:
        super().__init__()
        self._learning_rate = learning_rate
        self._rho = rho
        self._epsilon = epsilon

    def _initialize_history(
        self, parameter: ARRAY_TYPE
    ) -> Dict[str, ARRAY_TYPE]:
        """
        Initializes the history for the RMSProp optimizer.
        """
        backend, _ = get_backend()
        return {"accum_sq_grad": backend.zeros_like(parameter)}

    def optimize(
        self, history: Optional[Dict[str, ARRAY_TYPE]], derivative: ARRAY_TYPE
    ) -> Tuple[ARRAY_TYPE, Dict[str, ARRAY_TYPE]]:
        """
        Performs an optimization step using the RMSprop algorithm.

        Parameters
        ----------
        history : Optional[Dict[str, ARRAY_TYPE]]
            The history of the optimizer.
        derivative : ARRAY_TYPE
            The gradient of the loss function with respect to the parameters.

        Returns
        -------
        Tuple[ARRAY_TYPE, Dict[str, ARRAY_TYPE]]
            The parameter update and the updated history.
        """
        # Initialize history if it's None
        if history is None:
            history = self._initialize_history(derivative)

        backend, _ = get_backend()

        # Update the accumulated squared gradient with the current gradient
        accum_sq_grad = self._rho * history["accum_sq_grad"] + (
            1 - self._rho
        ) * backend.square(derivative)

        # Compute the parameter update using the RMSProp formula
        new_change = (self._learning_rate * derivative) / (
            backend.sqrt(accum_sq_grad) + self._epsilon
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
    The Adam optimizer is an adaptive learning rate optimization algorithm
    that's been designed specifically for training deep neural networks. It
    combines the advantages of two other extensions of stochastic gradient
    descent, namely AdaGrad and RMSProp.

    Parameters
    ----------
    learning_rate : float
        The learning rate or step size used for updating the parameters.
    beta_1 : float, optional
        The exponential decay rate for the first moment estimates
        (default is 0.9).
    beta_2 : float, optional
        The exponential decay rate for the second moment estimates
        (default is 0.999).
    epsilon : float, optional
        A small constant for numerical stability (default is 1e-07).
    """

    def __init__(
        self,
        learning_rate: float,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-08,
    ) -> None:
        super().__init__()
        self._learning_rate = learning_rate
        self._beta1 = beta_1
        self._beta2 = beta_2
        self._epsilon = epsilon

    def _initialize_history(
        self, parameter: ARRAY_TYPE
    ) -> Dict[str, ARRAY_TYPE]:
        """
        Initializes the history for the Adam optimizer.
        """
        backend, _ = get_backend()
        return {
            "first_moment_t": backend.zeros_like(parameter),
            "second_moment_t": backend.zeros_like(parameter),
        }

    def optimize(
        self, history: Optional[Dict[str, ARRAY_TYPE]], derivative: ARRAY_TYPE
    ) -> Tuple[ARRAY_TYPE, Dict[str, ARRAY_TYPE]]:
        """
        Performs an optimization step using the Adam algorithm.

        Parameters
        ----------
        history : Optional[Dict[str, ARRAY_TYPE]]
            The history of the optimizer.
        derivative : ARRAY_TYPE
            The gradient of the loss function with respect to the parameters.

        Returns
        -------
        Tuple[ARRAY_TYPE, Dict[str, ARRAY_TYPE]]
            The parameter update and the updated history.
        """
        # Get the current epoch
        epoch = self._epoch

        # Initialize history if it's None
        if history is None:
            history = self._initialize_history(derivative)

        # Retrieve previous first and second moment estimates from history
        first_moment_t_prev = history["first_moment_t"]
        second_moment_t_prev = history["second_moment_t"]

        backend, _ = get_backend()

        # Update biased first moment estimate
        first_moment_t = (
            self._beta1 * first_moment_t_prev + (1 - self._beta1) * derivative
        )
        # Update biased second raw moment estimate
        second_moment_t = (
            self._beta2 * second_moment_t_prev
            + (1 - self._beta2) * derivative * derivative
        )
        """
        Instead of using derivative * derivative, we could have used
        backend.square(derivative), which is essentially the same.
        However, there could be three ways of doing it:
            a. (1 - beta2) * backend.square(derivative)
            c. (1 - beta2) * (derivative * derivative)
            b. (1 - beta2) * derivative * derivative

        In a. & b., we are squaring the derivative first and then
        multiplying it by (1 - beta2). In b., we are multiplying
        (1 - beta2) with the derivative first and then again multiplying
        it with the derivative.

        They all should yield the same result, but since we are using floating
        point arithmetic, there could be slight differences in the results.
        For consistency(with frameworks), we are using the second approach (c)
        here.

        - May use same for RMSProp later.
        """
        # Compute bias-corrected first moment estimate
        first_mom_corr = 1 - self._beta1**epoch
        corrected_first_moment_t = first_moment_t / first_mom_corr

        # Compute bias-corrected second raw moment estimate
        second_mom_corr = 1 - self._beta2**epoch
        corrected_second_moment_t = second_moment_t / second_mom_corr

        # Compute the denominator for the update rule
        denom = backend.sqrt(corrected_second_moment_t) + self._epsilon

        # Compute the parameter update
        new_change = (self._learning_rate * corrected_first_moment_t) / denom
        # print("denom", self._learning_rate, self._beta1, epoch, first_mom_corr, self._learning_rate/first_mom_corr, first_moment_t, denom)

        # Store the updated first and second moment estimates in history
        new_history = {
            "first_moment_t": first_moment_t,
            "second_moment_t": second_moment_t,
        }

        return new_change, new_history


def get_optimizer(optimizer: str) -> Type[Optimizer]:
    """
    Returns the optimizer class based on the optimizer name.

    Parameters
    ----------
    optimizer : str
        The name of the optimizer.

    Returns
    -------
    Type[Optimizer]
        The optimizer class.
    """
    optimizer_map = {
        "sgd": SGD,
        "rmsprop": RMSProp,
        "adam": Adam,
    }
    return optimizer_map[optimizer]
