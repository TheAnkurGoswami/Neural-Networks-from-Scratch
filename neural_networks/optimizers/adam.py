from typing import Dict, Optional, Tuple

from neural_networks.backend import ARRAY_TYPE, get_backend
from neural_networks.optimizers.base import Optimizer


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.

    Adam is an optimization algorithm that computes adaptive learning rates for
    each parameter. It stores an exponentially decaying average of past squared
    gradients (like Adadelta and RMSprop) and an exponentially decaying average
    of past gradients (like momentum).
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        r"""
        Initializes the Adam optimizer.

        Args:
            learning_rate (float, optional): The learning rate (\(\eta\)).
                Defaults to 0.001.
            beta_1 (float, optional): The exponential decay rate for the first
                moment estimates (mean of gradients). Defaults to 0.9.
            beta_2 (float, optional): The exponential decay rate for the
                second moment estimates (variance of gradients).
                Defaults to 0.999.
            epsilon (float, optional): A small constant (\(\epsilon\)) for
                numerical stability, added to the denominator.
                Defaults to 1e-8.
        """
        super().__init__()
        if not 0.0 <= learning_rate:
            raise ValueError(
                f"Invalid learning rate: {learning_rate}, must be >= 0."
            )
        if not 0.0 <= beta_1 < 1.0:
            raise ValueError(f"Invalid beta_1: {beta_1}, must be in [0, 1).")
        if not 0.0 <= beta_2 < 1.0:
            raise ValueError(f"Invalid beta_2: {beta_2}, must be in [0, 1).")
        if not 0.0 < epsilon:
            raise ValueError(f"Invalid epsilon: {epsilon}, must be > 0.")

        self._learning_rate = learning_rate
        self._beta1 = beta_1
        self._beta2 = beta_2
        self._epsilon = epsilon
        # _epoch is the timestep, used for bias correction. Adam is stateful
        # per parameter update.
        # Note: self._epoch is set by set_cur_epoch and typically refers to
        # passes over the dataset.
        # Adam's epoch is more like an iteration counter.
        self._t: int = 0

    def _initialize_history(
        self, parameter: ARRAY_TYPE
    ) -> Dict[str, ARRAY_TYPE]:
        """
        Initializes history for Adam: first and second moment vectors.

        Args:
            parameter (ARRAY_TYPE): The parameter tensor, used for shape
                determination.

        Returns:
            Dict[str, ARRAY_TYPE]: Dictionary with "first_moment_t" (m_t) and
                "second_moment_t" (v_t) initialized to zeros.
        """
        backend, _ = get_backend()
        return {
            "first_moment_t": backend.zeros_like(parameter),  # m_0
            "second_moment_t": backend.zeros_like(parameter),  # v_0
        }

    def optimize(
        self, history: Optional[Dict[str, ARRAY_TYPE]], derivative: ARRAY_TYPE
    ) -> Tuple[ARRAY_TYPE, Dict[str, ARRAY_TYPE]]:
        r"""
        Performs a single optimization step using the Adam algorithm.

        Let \( g_t \) be the derivative (gradient) at timestep \( t \),
        \( \eta \) be the learning rate, \( \beta_1, \beta_2 \) be decay rates,
        \( m_t \) be the first moment vector (mean), and \( v_t \) be the
        second moment vector (uncentered variance).

        The update rules are:
        1. Update biased first moment estimate:
           .. math::
               m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
        2. Update biased second raw moment estimate:
           .. math::
               v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
        3. Compute bias-corrected first moment estimate:
           .. math::
               \hat{m}_t = \frac{m_t}{1 - \beta_1^t}
        4. Compute bias-corrected second raw moment estimate:
           .. math::
               \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
        5. Compute parameter update:
           .. math::
               \text{update_value} =
               \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}

        The parameter \( \theta \) is then updated as
        \( \theta_t = \theta_{t-1} - \text{update_value} \).

        Args:
            history (Optional[Dict[str, ARRAY_TYPE]]): Optimizer history,
                expects "first_moment_t" and "second_moment_t".
                Initialized if None.
            derivative (ARRAY_TYPE): The gradient \( g_t \).

        Returns:
            Tuple[ARRAY_TYPE, Dict[str, ARRAY_TYPE]]:
                - The calculated parameter update value.
                - The updated history dictionary with new "first_moment_t" and
                    "second_moment_t".
        """

        if history is None:
            history = self._initialize_history(derivative)

        # Retrieve previous first and second moment estimates from history
        first_moment_t_prev = history["first_moment_t"]
        second_moment_t_prev = history["second_moment_t"]

        backend, _ = get_backend()

        # Update biased first moment estimate
        # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        first_moment_t = (
            self._beta1 * first_moment_t_prev + (1 - self._beta1) * derivative
        )
        # Update biased second raw moment estimate
        # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        second_moment_t = (
            self._beta2 * second_moment_t_prev
            + (1 - self._beta2) * derivative * derivative
        )

        # NOTE:
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
        # m_hat_t = m_t / (1 - beta1^t)
        first_mom_corr = 1 - self._beta1**self._epoch
        corrected_first_moment_t = first_moment_t / first_mom_corr

        # Compute bias-corrected second raw moment estimate
        # v_hat_t = v_t / (1 - beta2^t)
        second_mom_corr = 1 - self._beta2**self._epoch
        corrected_second_moment_t = second_moment_t / second_mom_corr

        # Compute the parameter update
        denom = backend.sqrt(corrected_second_moment_t) + self._epsilon
        new_change = (self._learning_rate * corrected_first_moment_t) / denom

        # Store the updated first and second moment estimates in history
        new_history = {
            "first_moment_t": first_moment_t,
            "second_moment_t": second_moment_t,
        }

        return new_change, new_history
