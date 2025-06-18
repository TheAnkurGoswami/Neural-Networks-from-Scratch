from typing import Dict, List, Optional

import numpy as np
import torch as pt

from neural_networks.activations import Softmax
from neural_networks.backend import ARRAY_TYPE, get_backend
from neural_networks.optimizers import Optimizer


class ScaledDotProductAttention:
    def __init__(self, d_model: int, dim_k: int, dim_v: int) -> None:
        self._inputs: Optional[ARRAY_TYPE] = None
        self.parameters: List[str] = ["query", "key", "value"]
        self.weights: Dict[str, ARRAY_TYPE] = {}

        self.d_model: int = d_model
        _, backend_module = get_backend()
        if backend_module == "pt":
            self.dim_k: ARRAY_TYPE = pt.tensor(dim_k)
            self.dim_v: ARRAY_TYPE = pt.tensor(dim_v)
        elif backend_module == "np":
            self.dim_k: ARRAY_TYPE = np.array(dim_k, dtype=np.int32)
            self.dim_v: ARRAY_TYPE = np.array(dim_v, dtype=np.int32)
        self.parameter_dims: List[int] = [d_model, dim_k, dim_v]
        # For moving average based optimizers
        self._dW_history: Dict[str, Optional[Dict[str, ARRAY_TYPE]]] = {}
        self._build()

    def _build(self):
        """
        Initializes the weights for the self-attention mechanism.
        """
        _, backend_module = get_backend()
        for param, dim in zip(self.parameters, self.parameter_dims, strict=False):
            if backend_module == "pt":
                self.weights[param] = pt.normal(
                    mean=0.0, std=1.0, size=(self.d_model, dim)
                )
            elif backend_module == "np":
                self.weights[param] = np.random.normal(
                    loc=0.0,
                    scale=1.0,
                    size=(self.d_model, dim),
                ).astype(np.float32)

            self._dW_history[param] = None

    def forward(self, inputs):
        """
        Forward pass of the self-attention mechanism.
        """
        # Implement the forward pass logic here
        self.inputs = inputs
        #  Shape of inputs: (batch_size, seq_len, d_model)
        backend, _ = get_backend()
        self.projections = {}
        for param in self.parameters:
            self.projections[param] = backend.matmul(
                inputs, self.weights[param]
            )
        # self.Q = backend.matmul(
        #     inputs, self.weights["query"]
        # )
        # self.K = backend.matmul(
        #     inputs, self.weights["key"]
        # )
        # self.V = backend.matmul(
        #     inputs, self.weights["value"]
        # )
        """
        Q Shape : (batch_size, seq_len, d_model)
        K Shape : (batch_size, seq_len, dim_k)
        V Shape : (batch_size, seq_len, dim_v)
        """
        # Compute attention scores
        #  K.transpose(-1, -2) Shape : (batch_size, dim_k, seq_len)
        attention_scores = backend.matmul(
            self.projections["query"],
            self.projections["key"].transpose(-1, -2),
        )
        # For large values of dim_k, the dot products grow large in magnitude,
        # pushing the softmax function into regions where it has extremely
        # small gradients. To counteract this effect, we scale the dot products
        # by 1/√dim_k
        #  - Attention is all you need
        attention_scores /= backend.sqrt(self.dim_k)
        # print("Cust Attention Scores", attention_scores.shape, attention_scores)
        # attention_scores Shape : (batch_size, seq_len, seq_len)

        # Apply softmax to get attention weights
        # print("Attention bf softmax", attention_scores.shape, attention_scores)
        batch_dim, seq_len_dim, _ = attention_scores.shape
        attention_scores = attention_scores.reshape(
            (batch_dim * seq_len_dim, seq_len_dim)
        )
        self.softmax = Softmax(do_clip=False)
        # We do not clipping here as we are not feeding the softmax output to
        # log function. No need to clip the values to get stable gradients.
        softmax_attn = self.softmax.forward(attention_scores)
        self.softmax_attn = softmax_attn.reshape(
            (batch_dim, seq_len_dim, seq_len_dim)
        )
        # print("Cust Softmax attn", self.softmax_attn.shape, self.softmax_attn)
        self.attention = backend.matmul(
            self.softmax_attn, self.projections["value"]
        )
        # Shape of attention: (batch_size, seq_len, dim_v)
        return self.attention

    def backprop(self, dA: ARRAY_TYPE, optimizer: Optimizer) -> ARRAY_TYPE:
        """
        Backward pass of the self-attention
        """
        # dA Shape: (batch_size, seq_len, dim_v)
        backend, _ = get_backend()
        dV = backend.matmul(self.softmax_attn.transpose(-1, -2), dA)
        """
        Why transpose(softmax_attn) * dA & not dA * softmax_attn or
        softmax_attn * dA?

        Shape of dA: (batch_size, seq_len, dim_v)
        Shape of softmax_attn: (batch_size, seq_len, seq_len)
        It may seem like both will be a valid matrix multiplication as the
        resulting matrix will be of shape (batch_size, seq_len, dim_v), which
        corresponds to the shape of dV. However, the order of multiplication
        matters. To illustrate this, let's consider the shapes of the
        matrices:
        Shape of softmax_attn: (batch_size, S1, S2)
        Shape of V: (batch_size, S2, dim_v)
        Hence, Shape of dA: (batch_size, S1, dim_v)

        Now testing both of the cases above:
        Case 1: transpose(softmax_attn) * dA
        Shape of transpose(softmax_attn): (batch_size, S2, S1)
        transpose(softmax_attn) * dA will output a matrix of shape
        (batch_size, S2, dim_v).

        Case 2: dA * softmax_attn
        dA * softmax_attn will be an invalid matrix multiplication as the
        shapes of the matrices are not compatible.

        Case 1 is the correct one as it allows us to compute the gradient of
        the attention scores with respect to the input.
        """

        dW_v = backend.matmul(self.inputs.transpose(-1, -2), dV)
        dW_v = backend.sum(dW_v, axis=0)
        d_softmax_attention = backend.matmul(
            dA, self.projections["value"].transpose(-1, -2)
        )
        # print("d_softmax_attention shape", d_softmax_attention.shape, d_softmax_attention)
        batch_size, seq_len, _ = d_softmax_attention.shape
        d_softmax_attention = d_softmax_attention.reshape(
            (batch_size * seq_len, seq_len)
        )

        d_attention_scores = self.softmax.backprop(d_softmax_attention)
        d_attention_scores = d_attention_scores.reshape(
            (batch_size, seq_len, seq_len)
        )
        d_attention_scores /= backend.sqrt(self.dim_k)
        # print("d_attention_scores shape", d_attention_scores.shape, d_attention_scores)
        dQ = backend.matmul(d_attention_scores, self.projections["key"])
        dW_q = backend.matmul(self.inputs.transpose(-1, -2), dQ)
        dW_q = backend.sum(dW_q, axis=0)
        dK = backend.matmul(
            d_attention_scores.transpose(-1, -2), self.projections["query"]
        )
        dW_k = backend.matmul(self.inputs.transpose(-1, -2), dK)
        dW_k = backend.sum(dW_k, axis=0)
        # gradient w.r.t. X from each branch:
        dX_from_Q = backend.matmul(dQ, self.weights["query"].T)
        dX_from_K = backend.matmul(dK, self.weights["key"].T)
        dX_from_V = backend.matmul(dV, self.weights["value"].T)
        # print("dW_q shape", dW_q.shape, dW_q)
        # print("dW_k shape", dW_k.shape, dW_k)
        # print("dW_v shape", dW_v.shape, dW_v)
        for param, dW in zip(self.parameters, [dW_q, dW_k, dW_v], strict=False):
            dW_change, self._dW_history[param] = optimizer.optimize(
                self._dW_history[param], dW
            )
            # Parametric updates
            # print("dW_change shape", dW_change.shape, dW_change)
            # print("weights shape", self.weights[param].shape, self.weights[param])
            self.weights[param] -= dW_change

        # total gradient into X is the element‐wise sum:
        return dX_from_Q + dX_from_K + dX_from_V
