from typing import Dict, Optional, Tuple

from neural_networks.backend import ARRAY_TYPE, get_backend
from neural_networks.core.optimizer_base import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    Optionally includes momentum.
    """

    def __init__(self, learning_rate: float, momentum: float = 0.0) -> None:
        r"""
        Initializes the SGD optimizer.

        Args:
            learning_rate (float): The step size for parameter updates.
            momentum (float, optional): The momentum factor (\(\beta\)). Helps accelerate SGD
                                         in the relevant direction and dampens oscillations.
                                         Defaults to 0.0 (no momentum).
        """
        super().__init__()
        if not 0.0 <= learning_rate:
            raise ValueError(f"Invalid learning rate: {learning_rate}, must be >= 0.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}, must be in [0, 1).")

        self._learning_rate = learning_rate
        self._momentum = momentum

    def _initialize_history(
        self, parameter: ARRAY_TYPE
    ) -> Dict[str, ARRAY_TYPE]:
        """
        Initializes history for SGD, specifically the accumulated gradient for momentum.

        Args:
            parameter (ARRAY_TYPE): The parameter tensor, used to determine the shape
                                   for the history variable.

        Returns:
            Dict[str, ARRAY_TYPE]: A dictionary containing the initialized accumulated
                                   gradient ("accum_grad") as a zero tensor of the same
                                   shape as `parameter`.
        """
        backend, _ = get_backend()
        # 'accum_grad' stores the velocity term v_t for momentum
        return {"accum_grad": backend.zeros_like(parameter)}

    def optimize(
        self, history: Optional[Dict[str, ARRAY_TYPE]], derivative: ARRAY_TYPE
    ) -> Tuple[ARRAY_TYPE, Dict[str, ARRAY_TYPE]]:
        r"""
        Performs a single optimization step using SGD with optional momentum.

        Let \( g_t \) be the derivative (gradient) at timestep \( t \),
        \( \eta \) be the learning rate, and \( \beta \) be the momentum factor.

        If momentum (\( \beta > 0 \)):
        .. math::
            v_t = \beta v_{t-1} + g_t \\
            \text{update_value} = \eta v_t

        If no momentum (\( \beta = 0 \)):
        .. math::
            \text{update_value} = \eta g_t

        The parameter \( \theta \) is then updated as \( \theta_t = \theta_{t-1} - \text{update_value} \).

        Args:
            history (Optional[Dict[str, ARRAY_TYPE]]): Optimizer history for the parameter.
                Expected to contain "accum_grad" for momentum. Initialized if None.
            derivative (ARRAY_TYPE): The gradient of the loss with respect to the parameter
                                     (\( g_t \)).

        Returns:
            Tuple[ARRAY_TYPE, Dict[str, ARRAY_TYPE]]:
                - The calculated parameter update value (\( \eta v_t \) or \( \eta g_t \)).
                - The updated history dictionary with the new "accum_grad" (\( v_t \)).
        """
        if history is None:
            history = self._initialize_history(derivative)

        # Update the accumulated gradient with the current gradient
        accum_grad = self._momentum * history["accum_grad"] + derivative

        # Store the updated accumulated gradient in history
        new_history = {"accum_grad": accum_grad}
        return self._learning_rate * accum_grad, new_history
