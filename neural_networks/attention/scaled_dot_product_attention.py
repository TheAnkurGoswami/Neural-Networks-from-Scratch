from typing import Dict

import numpy as np
import torch as pt

from neural_networks.activations import Softmax
from neural_networks.attention.projection import Projection
from neural_networks.backend import ARRAY_TYPE, get_backend


class ScaledDotProductAttention:

    proj_layer: Dict[str, Projection] = {}
    dim_k = None

    def forward(self, q_proj, k_proj, v_proj):
        """
        Forward pass of the self-attention mechanism.
        """
        self.projections = {
            "query": q_proj,
            "key": k_proj,
            "value": v_proj,
        }
        """
        Q Shape : (batch_size, seq_len, d_model or dim_q)
        K Shape : (batch_size, seq_len, dim_k)
        V Shape : (batch_size, seq_len, dim_v)
        """
        backend, backend_module = get_backend()
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
        self.dim_k = self.projections["key"].shape[-1]
        if backend_module == "pt":
            self.dim_k: ARRAY_TYPE = pt.tensor(self.dim_k)
        elif backend_module == "np":
            self.dim_k: ARRAY_TYPE = np.array(self.dim_k, dtype=np.int32)
        attention_scores /= backend.sqrt(self.dim_k)
        # attention_scores Shape : (batch_size, seq_len, seq_len)
        # Apply softmax to get attention weights
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
        self.attention = backend.matmul(
            self.softmax_attn, self.projections["value"]
        )
        # Shape of attention: (batch_size, seq_len, dim_v)
        return self.attention

    def backprop(self, dA: ARRAY_TYPE) -> ARRAY_TYPE:
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

        d_softmax_attention = backend.matmul(
            dA, self.projections["value"].transpose(-1, -2)
        )
        batch_size, seq_len, _ = d_softmax_attention.shape
        d_softmax_attention = d_softmax_attention.reshape(
            (batch_size * seq_len, seq_len)
        )

        d_attention_scores = self.softmax.backprop(d_softmax_attention)
        d_attention_scores = d_attention_scores.reshape(
            (batch_size, seq_len, seq_len)
        )
        d_attention_scores /= backend.sqrt(self.dim_k)

        dQ = backend.matmul(d_attention_scores, self.projections["key"])

        dK = backend.matmul(
            d_attention_scores.transpose(-1, -2), self.projections["query"]
        )
        return dQ, dK, dV
