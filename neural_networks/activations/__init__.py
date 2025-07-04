from typing import Dict, Optional, Type

from neural_networks.activations.base import Activation

from .identity import Identity
from .relu import ReLU
from .sigmoid import Sigmoid
from .softmax import Softmax
from .tanh import Tanh

__all__ = ["Identity", "ReLU", "Sigmoid", "Softmax", "Tanh"]


# Helper to map activation strings to classes
_activation_map: Dict[str, Type[Activation]] = {
    "identity": Identity,
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "softmax": Softmax,
    "tanh": Tanh,
}


def get_activation_fn(activation_name: Optional[str]) -> Activation:
    """
    Retrieves an instance of an activation function by its name.
    If activation_name is None or "identity", returns an Identity activation
    instance.
    """
    if activation_name is None:
        activation_name = "identity"

    activation_class = _activation_map.get(activation_name.lower())
    if activation_class is None:
        raise ValueError(
            f"Unknown activation function: {activation_name}. "
            f"Available: {list(_activation_map.keys())}"
        )
    return activation_class
