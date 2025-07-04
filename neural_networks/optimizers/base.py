from typing import Dict, Optional, Tuple

from neural_networks.backend import ARRAY_TYPE


class Optimizer:
    """
    Base class for all optimizers.

    Optimizers are responsible for updating model parameters based on the
    gradients computed during backpropagation.
    """

    def __init__(self) -> None:
        """Initializes the optimizer."""
        self._epoch = 0
        # Current epoch, can be used by adaptive learning rate schedulers or
        # optimizers like Adam

    def _initialize_history(
        self, parameter: ARRAY_TYPE
    ) -> Dict[str, ARRAY_TYPE]:
        """
        Initializes any history variables required by the optimizer for a given
        parameter.

        This method is typically called by the `optimize` method the first time
        it encounters a new parameter or if history is not provided.
        Subclasses should override this to create structures like moving
        averages of gradients or squared gradients (e.g., for Adam or RMSProp).

        Args:
            parameter (ARRAY_TYPE): The parameter tensor for which to
                initialize history. Used to get the shape and device for
                history tensors.

        Returns:
            Dict[str, ARRAY_TYPE]: A dictionary containing initialized history
            tensors. Keys are specific to the optimizer implementation.
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
        r"""
        Performs a single optimization step and returns the parameter update.

        Subclasses must implement this method to define their specific
        optimization logic.
        The general update rule is often:
        \( \theta_{new} = \theta_{old} - \text{update_value} \)
        This method calculates `update_value`.

        Args:
            history (Optional[Dict[str, ARRAY_TYPE]]): A dictionary containing
                history tensors for the parameter being optimized
                (e.g., momentum, squared gradients). If None, it will be
                initialized by `_initialize_history`.
            derivative (ARRAY_TYPE): The gradient of the loss w.r.t the
                parameter that needs to be updated
                (\( \frac{\partial L}{\partial \theta} \)).

        Returns:
            Tuple[ARRAY_TYPE, Dict[str, ARRAY_TYPE]]:
                - The calculated parameter update value
                    (e.g., learning_rate * adjusted_gradient).
                - The updated history dictionary.
        """
        raise NotImplementedError()
