from neural_networks.backend import ARRAY_TYPE, NUMERIC_TYPE


class Loss:
    def forward(self, y_pred: ARRAY_TYPE, y_true: ARRAY_TYPE) -> NUMERIC_TYPE:
        r"""
        Computes the forward pass of the loss function.

        This method should be implemented by subclasses to define specific
        loss calculations.

        Args:
            y_pred (ARRAY_TYPE): Predicted values from the model.
            y_true (ARRAY_TYPE): True target values.

        Returns:
            NUMERIC_TYPE: The computed loss value.
        """
        raise NotImplementedError()

    def backprop(self) -> ARRAY_TYPE:
        r"""
        Computes the backward pass of the loss function.

        This method should be implemented by subclasses to calculate the
        gradient of the loss w.r.t the predicted values (\( y_{pred} \)).
        Typically, this is \( \frac{\partial L}{\partial y_{pred}} \).

        Returns:
            ARRAY_TYPE: Gradient of the loss w.r.t the predictions.
        """
        raise NotImplementedError()
