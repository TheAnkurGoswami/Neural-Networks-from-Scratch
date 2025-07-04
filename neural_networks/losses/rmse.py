from typing import Optional

from neural_networks.backend import ARRAY_TYPE, NUMERIC_TYPE, get_backend
from neural_networks.losses.mse import MSELoss


class RMSELoss(MSELoss):
    def __init__(self) -> None:
        """
        Initializes the Root Mean Squared Error (RMSE) Loss function.
        RMSE is the square root of MSE.

        Attributes:
            _loss (Optional[ARRAY_TYPE]): The RMSE loss value computed in the
                forward pass. Inherits _y_true, _y_pred, _size from MSELoss.
        """
        super().__init__()
        # Initialize MSELoss attributes like _y_true, _y_pred, _size
        self._loss: Optional[NUMERIC_TYPE] = None  # Loss is a numeric type

    def forward(self, y_pred: ARRAY_TYPE, y_true: ARRAY_TYPE) -> NUMERIC_TYPE:
        r"""
        Computes the Root Mean Squared Error (RMSE) loss.

        The RMSE is calculated as the square root of the Mean Squared Error
        (MSE):
        .. math::
            L_{RMSE} = \sqrt{L_{MSE}} =
                \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_{pred_i} - y_{true_i})^2}

        where:
         - \(N\) is the total number of elements
            (inherited from MSELoss as `_size`).
         - \(y_{pred_i}\) is the i-th predicted value.
         - \(y_{true_i}\) is the i-th true value.

        Args:
            y_pred (ARRAY_TYPE): Predicted values from the model.
            y_true (ARRAY_TYPE): True target values.

        Returns:
            NUMERIC_TYPE: The computed root mean squared error loss.
        """
        backend, _ = get_backend()
        mse_loss = super().forward(
            y_pred, y_true
        )  # This sets _y_pred, _y_true, _size in parent

        # Add a small epsilon for numerical stability if mse_loss can be zero.
        epsilon = 1e-12
        self._loss = backend.sqrt(mse_loss + epsilon)
        assert self._loss is not None, "RMSE loss calculation failed."
        return self._loss

    def backprop(self) -> ARRAY_TYPE:
        r"""
        Computes the gradient of the RMSE loss w.r.t the predictions
        (\( y_{pred} \)).

        The gradient is derived using the chain rule:
            ∂L/∂y_pred = (1 / (2 * sqrt(MSE))) * ∂(MSE)/∂y_pred
                => (1 / (2 * RMSE)) * ∂MSE/∂y_pred

        \( L_{RMSE} = \sqrt{L_{MSE}} \)
        .. math::
            \frac{\partial L_{RMSE}}{\partial y_{pred}} =
            \frac{\partial L_{RMSE}}{\partial L_{MSE}} \times
            \frac{\partial L_{MSE}}{\partial y_{pred}}

        Since \( \frac{\partial L_{RMSE}}{\partial L_{MSE}} =
            \frac{1}{2 \sqrt{L_{MSE}}} = \frac{1}{2 L_{RMSE}} \),
        the gradient becomes:
        .. math::
            \frac{\partial L_{RMSE}}{\partial y_{pred}} =
                \frac{1}{2 L_{RMSE}} \frac{\partial L_{MSE}}{\partial y_{pred}}

        This uses the `backprop` method of the parent `MSELoss` class.

        Returns:
            ARRAY_TYPE: Gradient of the RMSE loss w.r.t \( y_{pred} \).
        """
        # Attributes _y_pred, _y_true, _size are set by super().forward() call
        assert (
            self._y_pred is not None
        ), "y_pred not found. Forward pass must be called before backprop."
        assert (
            self._y_true is not None
        ), "y_true not found. Forward pass must be called before backprop."
        assert (
            self._loss is not None
        ), "RMSE loss (_loss) not computed. Forward pass must be called."

        mse_backprop = super().backprop()  # This is d(MSE)/dy_pred

        # Avoid division by zero if self._loss (RMSE) is very close to zero.
        # A small epsilon is added to the denominator for numerical stability.
        epsilon = 1e-12
        # Should be consistent with forward pass if used there for sqrt

        return (1 / (2 * (self._loss + epsilon))) * mse_backprop
