from typing import Dict, Optional, Tuple

from neural_networks.backend import ARRAY_TYPE, get_backend
from neural_networks.core.optimizer_base import Optimizer


class RMSProp(Optimizer):
    """
    RMSProp optimizer.

    RMSProp is an adaptive learning rate method that divides the learning rate
    for a weight by a running average of the magnitudes of recent gradients for
    that weight. It maintains a moving (discounted) average of the square of
    gradients and divides the gradient by the root of this average.
    """

    def __init__(
        self, learning_rate: float, rho: float = 0.9, epsilon: float = 1e-7
    ) -> None:
        r"""
        Initializes the RMSProp optimizer.

        Args:
            learning_rate (float): The learning rate (\(\eta\)).
            rho (float, optional): The discounting factor for the moving average of squared gradients (\(\rho\)).
                                   Defaults to 0.9.
            epsilon (float, optional): A small constant (\(\epsilon\)) added to the denominator for
                                       numerical stability. Defaults to 1e-7.
        """
        super().__init__()
        if not 0.0 <= learning_rate:
            raise ValueError(f"Invalid learning rate: {learning_rate}, must be >= 0.")
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"Invalid rho (discounting factor): {rho}, must be in [0, 1).")
        if not 0.0 < epsilon:
            raise ValueError(f"Invalid epsilon: {epsilon}, must be > 0.")

        self._learning_rate = learning_rate
        self._rho = rho
        self._epsilon = epsilon

    def _initialize_history(
        self, parameter: ARRAY_TYPE
    ) -> Dict[str, ARRAY_TYPE]:
        """
        Initializes history for RMSProp: the accumulated squared gradients.

        Args:
            parameter (ARRAY_TYPE): The parameter tensor, used for shape determination.

        Returns:
            Dict[str, ARRAY_TYPE]: Dictionary with "accum_sq_grad" initialized to zeros.
        """
        backend, _ = get_backend()
        # 'accum_sq_grad' stores E[g^2]_t, the moving average of squared gradients.
        return {"accum_sq_grad": backend.zeros_like(parameter)}

    def optimize(
        self, history: Optional[Dict[str, ARRAY_TYPE]], derivative: ARRAY_TYPE
    ) -> Tuple[ARRAY_TYPE, Dict[str, ARRAY_TYPE]]:
        r"""
        Performs a single optimization step using the RMSProp algorithm.

        Let \( g_t \) be the derivative (gradient) at timestep \( t \),
        \( \eta \) be the learning rate, \( \rho \) be the discounting factor,
        and \( E[g^2]_t \) be the moving average of squared gradients.

        The update rule is:
        .. math::
            E[g^2]_t = \rho E[g^2]_{t-1} + (1 - \rho) g_t^2 \\
            \text{update_value} = \frac{\eta}{\sqrt{E[g^2]_t} + \epsilon} g_t

        The parameter \( \theta \) is then updated as \( \theta_t = \theta_{t-1} - \text{update_value} \).

        Args:
            history (Optional[Dict[str, ARRAY_TYPE]]): Optimizer history, expects "accum_sq_grad".
                                                     Initialized if None.
            derivative (ARRAY_TYPE): The gradient \( g_t \).

        Returns:
            Tuple[ARRAY_TYPE, Dict[str, ARRAY_TYPE]]:
                - The calculated parameter update value.
                - The updated history dictionary with new "accum_sq_grad".
        """
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
        # self._epoch += 1 # Epoch is managed by the training loop, optimizer should not increment it here.
                           # It's set via set_cur_epoch.

        return new_change, new_history
