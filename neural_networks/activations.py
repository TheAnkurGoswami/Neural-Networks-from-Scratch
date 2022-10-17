

from typing import Optional
import numpy as np


class Activation:
    @staticmethod
    def forward(inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    @staticmethod
    def backprop(dA: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class Identity(Activation):
    @staticmethod
    def forward(inputs: np.ndarray) -> np.ndarray:
        return inputs

    @staticmethod
    def backprop(dA: np.ndarray) -> np.ndarray:
        return dA


class ReLU(Activation):
    @staticmethod
    def forward(inputs: np.ndarray) -> np.ndarray:
        return np.where(inputs > 0, inputs, 0)

    @staticmethod
    def backprop(dA: np.ndarray) -> np.ndarray:
        return dA


def get_activation_fn(activation: Optional[str]) -> Activation:
    if activation is None:
        activation = "identity"
    activation_map = {
        "identity": Identity,
        "relu": ReLU,
    }
    return activation_map[activation]