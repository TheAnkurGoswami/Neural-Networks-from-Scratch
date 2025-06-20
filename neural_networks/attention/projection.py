
import numpy as np
import torch as pt

from neural_networks.backend import get_backend
from neural_networks.nn import Dense
from neural_networks.optimizers import Optimizer


class Projection(Dense):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        add_bias: bool = True,
        **kwargs
    ) -> None:
        """
        Projection layer is linear transformation (W.X + B).
        This is same as Dense layer, however the backpropagation is a bit
        different. To keep things clean & straightforward, this separate
        module/class is defined.
        """

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            add_bias=add_bias,
            **kwargs
        )

        _, backend_module = get_backend()
        if self.add_bias:
            if backend_module == "pt":
                self._bias = pt.zeros(size=(1, 1, out_features))
            elif backend_module == "np":
                self._bias = np.zeros(shape=(1, 1, out_features)).astype(
                    np.float32
                )

    def backprop(self, dZ, optimizer: Optimizer):
        backend, _ = get_backend()
        dW = backend.matmul(self._inputs.transpose(-1, -2), dZ)
        dW = backend.sum(dW, axis=0)
        dw_change, self._dw_history = optimizer.optimize(self._dw_history, dW)
        self._weights -= dw_change

        dX = backend.matmul(dZ, self._weights.T)

        if self.add_bias:
            dB = backend.sum(dZ, axis=0, keepdims=True)
            dB = backend.sum(dB, axis=1, keepdims=True)
            db_change, self._db_history = optimizer.optimize(
                self._db_history, dB
            )
            self._bias -= db_change
        else:
            dB = None

        if self._retain_grad:
            self._dW = dW
            self._dB = dB
            self._dZ = dZ

        return dX
