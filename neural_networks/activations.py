from typing import Optional, Type

import numpy as np


class Activation:
    """
    Base class for activation functions.
    """

    def __init__(self) -> None:
        self._input: np.ndarray = np.array([])
        self._activation: np.ndarray = np.array([])

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass of the activation function.

        Args:
            inputs (np.ndarray): Input data.

        Returns:
            np.ndarray: Output after applying the activation function.
        """
        self._input = inputs
        # Below statement does nothing, just for the type matching
        self._activation = np.zeros_like(inputs)
        return self._activation

    def derivative(self) -> np.ndarray:
        """
        Compute the derivative of the activation function.

        Returns:
            np.ndarray: Derivative of the activation function.
        """
        raise NotImplementedError()

    def backprop(self, dA: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass of the activation function.

        Args:
            dA (np.ndarray): Gradient of the loss with respect to the output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        return dA * self.derivative()


class Identity(Activation):
    """
    Identity activation function.
    """

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        super().forward(inputs)
        return inputs

    def derivative(self) -> np.ndarray:
        return np.ones_like(self._input)


class ReLU(Activation):
    """
    ReLU activation function.
    """

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        super().forward(inputs)
        return np.where(inputs > 0, inputs, 0)

    def derivative(self) -> np.ndarray:
        return np.where(self._input > 0, 1, 0)


class Sigmoid(Activation):
    """
    Sigmoid activation function.
    """

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._activation = 1 / (1 + np.exp(-1 * inputs))
        return self._activation

    def derivative(self) -> np.ndarray:
        """
        Compute the derivative of the sigmoid function.

        Returns:
            np.ndarray: Derivative of the sigmoid function.
        """
        return self._activation * (1 - self._activation)


class Tanh(Activation):
    """
    Tanh activation function.
    """

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._activation = np.tanh(inputs)
        return self._activation

    def derivative(self) -> np.ndarray:
        """
        Compute the derivative of the tanh function.

        Returns:
            np.ndarray: Derivative of the tanh function.
        """
        return 1 - np.square(self._activation)


class Softmax(Activation):
    def forward(self, inputs: np.ndarray):
        r"""
        inputs -> Z
        \hat{y}_{i} = \frac{e^{z_{i}}}{\sum_{j} e^{z_{j}}}
        output -> Y_hat
        """
        num = np.exp(inputs)
        denom = np.sum(num)
        self._activation = num / denom
        return self._activation

    def derivative(self):
        r"""
        \frac{d\hat{y}_{i}}{dz_{i}} = \hat{y}_{i} (1 - \hat{y}_{i})
        \frac{d\hat{y}_{i}}{dz_{j}} = -\hat{y}_{i} \hat{y}_{j}

        \frac{d\hat{Y}}{dZ} = J_{ij}
        \text{J}_{ij} = \begin{cases}
            \hat{y}_{i} (1 - \hat{y}_{i}) & \text{if } i=j,
            \\ -\hat{y}_{i} \hat{y}_{j} & \text{if } i \neq j
            \end{cases}
        """
        jacobian_mat = np.zeros(
            (self._activation.shape[1], self._activation.shape[1])
        )
        for row_idx in range(self._activation.shape[1]):
            for col_idx in range(row_idx, self._activation.shape[1]):
                if row_idx == col_idx:
                    jacobian_mat[row_idx, col_idx] = self._activation[0][
                        row_idx
                    ] * (1 - self._activation[0][row_idx])
                else:
                    jacobian_mat[row_idx, col_idx] = jacobian_mat[
                        col_idx, row_idx
                    ] = (
                        -self._activation[0][row_idx]
                        * self._activation[0][col_idx]
                    )

        # print(jacobian_mat)
        return jacobian_mat

    def backprop(self, dA: np.ndarray):
        """
        dA = dL/da = dY_hat
        """
        jac_mat = self.derivative()
        dZ = np.matmul(dA, jac_mat)
        return dZ


def get_activation_fn(activation: Optional[str]) -> Type[Activation]:
    """
    Get the activation function class based on the activation name.

    Args:
        activation (Optional[str]): Name of the activation function.

    Returns:
        Type[Activation]: Activation function class.
    """
    if activation is None:
        activation = "identity"
    activation_map = {
        "identity": Identity,
        "relu": ReLU,
        "sigmoid": Sigmoid,
        "tanh": Tanh,
        "softmax": Softmax,
    }
    return activation_map[activation]
