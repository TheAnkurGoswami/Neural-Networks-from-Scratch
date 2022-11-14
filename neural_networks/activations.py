

from typing import Optional, Type

import numpy as np


class Activation:
    @staticmethod
    def forward(inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def backprop(dA: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class Identity(Activation):
    @staticmethod
    def forward(inputs: np.ndarray) -> np.ndarray:
        return inputs

    @staticmethod
    def backprop(dA: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        return dA


class ReLU(Activation):
    @staticmethod
    def forward(inputs: np.ndarray) -> np.ndarray:
        return np.where(inputs > 0, inputs, 0)

    @staticmethod
    def backprop(dA: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        return dA


class Sigmoid(Activation):
    @staticmethod
    def forward(inputs: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-1 * inputs))

    @staticmethod
    def backprop(dA: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        y = sigmoid(x) = 1 / (1 + e^(-x))
        dy/dx = sigmoid(x) * (1 - sigmoid(x))
        """
        return dA * Sigmoid.forward(inputs) * (1 - Sigmoid.forward(inputs))


class Tanh(Activation):
    @staticmethod
    def forward(inputs: np.ndarray) -> np.ndarray:
        return np.tanh(inputs)

    @staticmethod
    def backprop(dA: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        y = tanh(x)
        dy/dx = 1 - (tanh(x))^2
        """
        return dA * (1 - np.square(np.tanh(inputs)))


def get_activation_fn(activation: Optional[str]) -> Type[Activation]:
    if activation is None:
        activation = "identity"
    activation_map = {
        "identity": Identity,
        "relu": ReLU,
    }
    return activation_map[activation]
