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
        """
        Performs the forward pass for the activation function.

        For the Identity function, the output is the same as the input.
        The input is stored as the activation output.

        Args:
            inputs (np.ndarray): The input data to the activation function.

        Returns:
            np.ndarray: The output after applying the activation function.
        """
        self._activation = inputs

        super().forward(inputs)
        return inputs

    def derivative(self) -> np.ndarray:
        """
        Computes the derivative of the Identity activation function.

        The derivative of the Identity function is always 1.

        Returns:
            np.ndarray: An array of ones with the same shape as the input,
            representing the derivative of the activation function.
        """

        return np.ones_like(self._input)


class ReLU(Activation):
    """
    ReLU activation function.
    """

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Applies the ReLU (Rectified Linear Unit) activation function to the
        input data.
        The ReLU function is defined as:
            f(x) = max(0, x)
        Parameters:
            inputs (np.ndarray): The input data, typically a NumPy array of any
            shape.
        Returns:
            np.ndarray: The output data after applying the ReLU activation
            function,
                        with the same shape as the input.
        Formula:
            output = np.where(inputs > 0, inputs, 0)
        """

        super().forward(inputs)
        return np.where(inputs > 0, inputs, 0)

    def derivative(self) -> np.ndarray:
        """
        Computes the derivative of the activation function.
        For the ReLU (Rectified Linear Unit) activation function, the
        derivative is defined as:
            f'(x) = 1 if x > 0
                    0 if x <= 0
        Returns:
            np.ndarray: An array containing the derivative values for the
            input.
        """

        return np.where(self._input > 0, 1, 0)


class Sigmoid(Activation):
    """
    Sigmoid activation function.
    """

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the activation function.
        This method applies the sigmoid activation function to the input data.
        The sigmoid function is defined as:
            σ(x) = 1 / (1 + e^(-x))
        where `x` is the input.
        Parameters:
            inputs (np.ndarray): The input data as a NumPy array.
        Returns:
            np.ndarray: The output of the sigmoid activation function applied
            to the input.
        """

        self._activation = 1 / (1 + np.exp(-1 * inputs))
        return self._activation

    def derivative(self) -> np.ndarray:
        """
        Computes the derivative of the activation function.
        The derivative is calculated using the formula:
            f'(x) = f(x) * (1 - f(x))
        where f(x) is the activation value.
        Returns:
            np.ndarray: The derivative of the activation function.
        """

        return self._activation * (1 - self._activation)


class Tanh(Activation):
    """
    Tanh activation function.
    """

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass using the hyperbolic tangent (tanh)
        activation function.
        The tanh activation function is defined as:
            tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        Parameters:
            inputs (np.ndarray): The input data to the activation function.
        Returns:
            np.ndarray: The output after applying the tanh activation function.
        """

        self._activation = np.tanh(inputs)
        return self._activation

    def derivative(self) -> np.ndarray:
        """
        Computes the derivative of the activation function.
        Formula:
            f'(x) = 1 - tanh^2(x)
        Returns:
            np.ndarray: The derivative of the activation function.
        """

        return 1 - np.square(self._activation)


class Softmax(Activation):
    def forward(self, inputs: np.ndarray):
        """
        Performs the forward pass of the activation function.
        This method computes the activation values using the softmax function:
            softmax(x_i) = exp(x_i) / Σ(exp(x_j))
        where x_i is an input value, and the denominator is the sum of the
        exponentials of all input values.
        Args:
            inputs (np.ndarray): The input array for which the activation is
            computed.
        Returns:
            np.ndarray: The computed activation values as a normalized
            probability distribution.
        """

        num = np.exp(inputs)
        # print("inputs", inputs)
        denom = np.sum(num, axis=1, keepdims=True)
        self._activation = num / denom
        # print("self._activ", self._activation)
        return self._activation

    def derivative(self):
        r"""
        Computes the Jacobian matrix of the derivative of the activation
        function.

        The Jacobian matrix represents the partial derivatives of the output of
        the activation function with respect to its input. For a softmax
        activation function, the Jacobian matrix is defined as:

        .. math::
            \frac{\partial \hat{y}_{i}}{\partial z_{i}} =
                \hat{y}_{i} (1 - \hat{y}_{i})

            \frac{\partial \hat{y}_{i}}{\partial z_{j}} =
                -\hat{y}_{i} \hat{y}_{j}, \quad i \neq j

        The Jacobian matrix :math:`\frac{\partial \hat{Y}}{\partial Z}` is
        given by:

        .. math::
            J_{ij} =
            \begin{cases}
                \hat{y}_{i} (1 - \hat{y}_{i}) & \text{if } i = j, \\
                -\hat{y}_{i} \hat{y}_{j} & \text{if } i \neq j

        Returns:
            numpy.ndarray: The Jacobian matrix of shape (n, n), where n is the
            number of elements in the activation output.
        """
        jacobian_mat = np.zeros(
            (self._activation.shape[0], self._activation.shape[1], self._activation.shape[1])
        )
        for batch_idx in range(self._activation.shape[0]):
            for row_idx in range(self._activation.shape[1]):
                for col_idx in range(row_idx, self._activation.shape[1]):
                    if row_idx == col_idx:
                        jacobian_mat[batch_idx, row_idx, col_idx] = self._activation[batch_idx][
                            row_idx
                        ] * (1 - self._activation[batch_idx][row_idx])
                    else:
                        jacobian_mat[batch_idx, row_idx, col_idx] = jacobian_mat[
                            batch_idx, col_idx, row_idx
                        ] = (
                            -self._activation[batch_idx][row_idx]
                            * self._activation[batch_idx][col_idx]
                        )
        return jacobian_mat

    def backprop(self, dA: np.ndarray):
        """
        dA = dL/da = dY_hat
        """
        print("dA", dA.shape, dA)
        jac_mat = self.derivative()
        dZ_arr = []
        # print("jacobian shape", jac_mat.shape, dA.shape)
        # print("="*100)
        for batch_idx in range(jac_mat.shape[0]):
            # dZ = np.matmul(dA, jac_mat[batch_idx])
            # print("dA", dA.shape, dA)
            dZ = np.matmul(dA[[batch_idx], :], jac_mat[batch_idx])
            # print("jac_mat", jac_mat[batch_idx].shape, jac_mat[batch_idx])
            dZ_arr.append(dZ.flatten())
        # print("="*100)
        return np.array(dZ_arr)
        # print(np.array(dZ_arr))
        return (np.mean(dZ_arr, axis=0))
        dZ = np.matmul(dA, jac_mat)
        return dZ

# def batch_matrix_vector_multiply(matrix: np.ndarray, batch_matrices: np.ndarray) -> np.ndarray:
#     """
#     Multiplies each row of a matrix with the corresponding matrix in a batch of matrices.

#     Args:
#     matrix (np.ndarray): A 2D array of shape (r, m).
#     batch_matrices (np.ndarray): A 3D array of shape (r, m, m).

#     Returns:
#     np.ndarray: A 2D array of shape (r, m) resulting from the multiplication.
#     """
#     result = np.einsum('rm,rmm->rm', matrix, batch_matrices)
#     return result
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
