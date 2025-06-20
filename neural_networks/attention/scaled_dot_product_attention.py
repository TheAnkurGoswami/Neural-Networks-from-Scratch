from typing import Dict, List, Optional

import numpy as np
import torch as pt

from neural_networks.activations import Softmax
from neural_networks.attention.projection import Projection
from neural_networks.backend import ARRAY_TYPE, get_backend
from neural_networks.optimizers import Optimizer


class ScaledDotProductAttention:
    def __init__(
        self,
        d_model: int,
        dim_k: int,
        dim_v: int,
        dim_q: Optional[int] = None,
        add_bias: bool = True,
    ) -> None:
        self._inputs: Optional[ARRAY_TYPE] = None
        self.parameters: List[str] = ["query", "key", "value"]
        self.add_bias = add_bias
        self.proj_layer: Dict[str, Projection] = {}

        self.d_model: int = d_model
        _, backend_module = get_backend()
        if backend_module == "pt":
            self.d_model: ARRAY_TYPE = pt.tensor(d_model)
            if dim_q is not None:
                self.dim_q: ARRAY_TYPE = pt.tensor(dim_q)
            self.dim_k: ARRAY_TYPE = pt.tensor(dim_k)
            self.dim_v: ARRAY_TYPE = pt.tensor(dim_v)
        elif backend_module == "np":
            self.d_model: ARRAY_TYPE = np.array(d_model, dtype=np.int32)
            if dim_q is not None:
                self.dim_q: ARRAY_TYPE = np.array(dim_q, dtype=np.int32)
            self.dim_k: ARRAY_TYPE = np.array(dim_k, dtype=np.int32)
            self.dim_v: ARRAY_TYPE = np.array(dim_v, dtype=np.int32)

        self.parameter_dims: List[int] = [
            dim_q if hasattr(self, "dim_q") else d_model,
            dim_k,
            dim_v,
        ]
        # For moving average based optimizers
        self._build()

    def _build(self):
        """
        Initializes the weights for the self-attention mechanism.
        """
        for param, dim in zip(
            self.parameters, self.parameter_dims, strict=False
        ):
            self.proj_layer[param] = Projection(
                in_features=self.d_model,
                out_features=dim,
                add_bias=self.add_bias,
            )

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
            self.projections[param] = self.proj_layer[param].forward(inputs)
        """
        Q Shape : (batch_size, seq_len, d_model or dim_q)
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
        backend, backend_module = get_backend()
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

        dK = backend.matmul(
            d_attention_scores.transpose(-1, -2), self.projections["query"]
        )

        dX_from_Q = self.proj_layer["query"].backprop(dQ, optimizer)
        dX_from_K = self.proj_layer["key"].backprop(dK, optimizer)
        dX_from_V = self.proj_layer["value"].backprop(dV, optimizer)

        # total gradient into X is the element‐wise sum:
        return dX_from_Q + dX_from_K + dX_from_V
