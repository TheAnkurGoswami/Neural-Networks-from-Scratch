

from typing import Optional, Type

import numpy as np


class Activation:
    def __init__(self) -> None:
        self._input: np.ndarray = np.array([])

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self._input = inputs
        # Below statement does nothing, just for the type matching
        return np.zeros_like(inputs)

    def derivative(self) -> np.ndarray:
        raise NotImplementedError()

    def backprop(self, dA: np.ndarray) -> np.ndarray:
        print("dA", dA.shape)
        print("self.derivative", self.derivative().shape)
        return dA * self.derivative()


class Identity(Activation):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        super().forward(inputs)
        return inputs

    def derivative(self) -> np.ndarray:
        return np.ones_like(self._input)


class ReLU(Activation):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        super().forward(inputs)
        return np.where(inputs > 0, inputs, 0)

    def derivative(self) -> np.ndarray:
        return np.where(self._input > 0, 1, 0)


class Sigmoid(Activation):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        super().forward(inputs)
        return 1 / (1 + np.exp(-1 * inputs))

    def derivative(self) -> np.ndarray:
        """
        y = sigmoid(x) = 1 / (1 + e^(-x))
        dy/dx = sigmoid(x) * (1 - sigmoid(x))
        """
        return self.forward(self._input) * (1 - self.forward(self._input))


class Softmax(Activation):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        super().forward(inputs)
        return np.exp(inputs) / np.sum(np.exp(inputs))

    def derivative(self) -> np.ndarray:
        # FIXME
        """
        y = sigmoid(x) = 1 / (1 + e^(-x))
        dy/dx = sigmoid(x) * (1 - sigmoid(x))
        """

        kronecker_mask = np.eye(self._input.shape[1])
        activation = self.forward(self._input)
        print(activation.shape, kronecker_mask.shape, (kronecker_mask - activation).shape)
        return np.matmul((kronecker_mask - activation), activation.T)


class Tanh(Activation):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        super().forward(inputs)
        return np.tanh(inputs)

    def derivative(self) -> np.ndarray:
        """
        y = tanh(x)
        dy/dx = 1 - (tanh(x))^2
        """
        return (1 - np.square(np.tanh(self._input)))


def get_activation_fn(activation: Optional[str]) -> Type[Activation]:
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
