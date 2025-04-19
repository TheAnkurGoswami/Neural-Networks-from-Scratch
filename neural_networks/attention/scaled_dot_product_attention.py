from typing import Optional

import numpy as np
import torch as pt
from neural_networks.activations import Softmax
from neural_networks.backend import ARRAY_TYPE, get_backend

class ScaledDotProductAttention:
    def __init__(self, d_model: int, dim_q: int, dim_k: int, dim_v: int) -> None:
        self.query_w: Optional[ARRAY_TYPE] = None
        self.key_w: Optional[ARRAY_TYPE] = None
        self.value_w: Optional[ARRAY_TYPE] = None
        self.d_model: int = d_model
        _, backend_module = get_backend()
        if backend_module == "pt":
            self.dim_q: ARRAY_TYPE = pt.tensor(dim_q)
            self.dim_k: ARRAY_TYPE = pt.tensor(dim_k)
            self.dim_v: ARRAY_TYPE = pt.tensor(dim_v)
        elif backend_module == "np":
            self.dim_q: ARRAY_TYPE = np.array(dim_q, dtype=np.int32)
            self.dim_k: ARRAY_TYPE = np.array(dim_k, dtype=np.int32)
            self.dim_v: ARRAY_TYPE = np.array(dim_v, dtype=np.int32)
        self._build()

    def _build(self):
        """
        Initializes the weights for the self-attention mechanism.
        """
        _, backend_module = get_backend()
        if backend_module == "pt":
            self.query_w = pt.normal(
                mean=0.0, std=1.0, size=(self.d_model, self.dim_q)
            )
            self.key_w = pt.normal(
                mean=0.0, std=1.0, size=(self.d_model, self.dim_k)
            )
            self.value_w = pt.normal(
                mean=0.0, std=1.0, size=(self.d_model, self.dim_v)
            )
        elif backend_module == "np":
            self.query_w = np.random.normal(
                loc=0.0,
                scale=1.0,
                size=(self.d_model, self.dim_q),
            ).astype(np.float32)
            self.key_w = np.random.normal(
                loc=0.0,
                scale=1.0,
                size=(self.d_model, self.dim_k),
            ).astype(np.float32)
            self.value_w = np.random.normal(
                loc=0.0,
                scale=1.0,
                size=(self.d_model, self.dim_v),
            ).astype(np.float32)

    def forward(self, inputs):
        """
        Forward pass of the self-attention mechanism.
        """
        # Implement the forward pass logic here
        #  Shape of inputs: (batch_size, seq_len, d_model)
        backend, _ = get_backend()
        Q = backend.matmul(
            inputs, self.query_w
        )
        K = backend.matmul(
            inputs, self.key_w
        )
        V = backend.matmul(
            inputs, self.value_w
        )
        """
        Q Shape : (batch_size, seq_len, dim_q)
        K Shape : (batch_size, seq_len, dim_k)
        V Shape : (batch_size, seq_len, dim_v)
        """
        # Compute attention scores
        #  K.transpose(-1, -2) Shape : (batch_size, dim_k, seq_len)
        attention_scores = backend.matmul(Q, K.transpose(-1, -2))
        # For large values of dim_k, the dot products grow large in magnitude,
        # pushing the softmax function into regions where it has extremely
        # small gradients. To counteract this effect, we scale the dot products
        # by 1/√dim_k
        #  - Attention is all you need
        attention_scores /= backend.sqrt(self.dim_k)
        # attention_scores Shape : (batch_size, seq_len, seq_len)

        # Apply softmax to get attention weights
        # print("Attention bf softmax", attention_scores.shape, attention_scores)
        batch_dim, seq_len_dim, _ = attention_scores.shape
        attention_scores = attention_scores.reshape((batch_dim*seq_len_dim, seq_len_dim))
        self.softmax = Softmax()
        softmax_attn = self.softmax.forward(attention_scores)
        self.softmax_attn = softmax_attn.reshape((batch_dim, seq_len_dim, seq_len_dim))

        self.attention = backend.matmul(self.softmax_attn, V)
        # Shape of attention: (batch_size, seq_len, dim_v)
        return self.attention

    def backprop(self, dA):
        """
        Backward pass of the self-attention
        """
        # Implement the backward pass logic here
        # dV = dA * self.softmax_attn
        pass
