

import numpy as np


class Activation:
    @staticmethod
    def forward(inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    @staticmethod
    def backprop() -> np.ndarray:
        raise NotImplementedError()

class ReLU(Activation):
    @staticmethod
    def forward(inputs: np.ndarray) -> np.ndarray:
        return np.where(inputs > 0, inputs, 0)

    @staticmethod
    def backprop(dA: np.ndarray) -> np.ndarray:
        return dA


def get_activation_fn(activation: str) -> Activation:
    activation_map = {
        "relu": ReLU,
    }
    return activation_map[activation]