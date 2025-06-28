import os
from typing import Union

import numpy as np
import torch as pt

ARRAY_TYPE = Union[np.ndarray, pt.Tensor]
NUMERIC_TYPE = Union[float, int, np.number, pt.Tensor]


def get_backend():
    """
    Returns the backend module being used.
    """
    backend_module = os.getenv("BACKEND", default="pt")

    backend = np if backend_module == "np" else pt
    return backend, backend_module
