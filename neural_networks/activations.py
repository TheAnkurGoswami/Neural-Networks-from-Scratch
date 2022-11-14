

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
