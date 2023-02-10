from typing import Union

import numpy as np
from tensorflow import Tensor as TF_Tensor
from torch import Tensor as PT_Tensor

NUMBER_TYPE = Union[np.ndarray, float, int, PT_Tensor, TF_Tensor]


def check_closeness(
        a: NUMBER_TYPE,
        b: NUMBER_TYPE,
        double_check: bool = True,
        tolerance: float = 1e-06) -> bool:
    main_check = np.allclose(a, b)
    other_check = np.abs(a - b) < tolerance
    if double_check:
        return bool(main_check or np.all(other_check))
    return main_check


