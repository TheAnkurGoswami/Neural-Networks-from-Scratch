from typing import Union

import numpy as np

NUMBER_TYPE = Union[np.ndarray, float, int]

def check_closeness(
        a: NUMBER_TYPE,
        b: NUMBER_TYPE,
        double_check: bool = False,
        tolerance: float = 1e-06) -> bool:
    main_check = np.allclose(a, b)
    other_check = np.abs(a - b) < tolerance
    if double_check:
        return main_check or np.all(other_check)
    return main_check


