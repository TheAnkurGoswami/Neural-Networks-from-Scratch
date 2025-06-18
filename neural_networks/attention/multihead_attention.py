from typing import Optional

import numpy as np
import torch as pt

from neural_networks.attention.scaled_dot_product_attention import (
    ScaledDotProductAttention,
)
from neural_networks.backend import ARRAY_TYPE, get_backend


class MultiHeadAttention:
    def __init__(
        self, d_model: int, num_heads: int, dim_q: int, dim_k: int, dim_v: int
    ) -> None:
        self.output_w: Optional[ARRAY_TYPE] = None
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        _, backend_module = get_backend()
        if backend_module == "pt":
            self.d_model: ARRAY_TYPE = pt.tensor(d_model)
        elif backend_module == "np":
            self.d_model: ARRAY_TYPE = np.array(d_model, dtype=np.int32)

        self.dim_q = self.d_model // self.num_heads
        self.dim_k = self.d_model // self.num_heads
        self.dim_v = self.d_model // self.num_heads

        self._build()

    def _build(self):
        """
        Initializes the weights for the multi-head attention mechanism.
        """
        self.attn_heads = [
            ScaledDotProductAttention(
                d_model=self.d_model,
                dim_q=self.dim_q,
                dim_k=self.dim_k,
                dim_v=self.dim_v,
            )
            for _ in range(self.num_heads)
        ]
        _, backend_module = get_backend()
        if backend_module == "pt":
            self.output_w = pt.normal(
                mean=0.0,
                std=1.0,
                size=(self.num_heads * self.dim_v, self.d_model),
            )
        elif backend_module == "np":
            self.output_w = np.random.normal(
                loc=0.0,
                scale=1.0,
                size=(self.num_heads * self.dim_v, self.d_model),
            ).astype(np.float32)

    def forward(self, inputs):
        """
        Forward pass of the multi-head attention mechanism.
        """
        _, backend_module = get_backend()
        if backend_module == "pt":
            batch_size = inputs.shape[0]
            seq_len = inputs.shape[1]
            all_outputs = []
            for head in self.attn_heads:
                output = head.forward(inputs)
                all_outputs.append(output)
            concatenated_output = pt.cat(all_outputs, dim=2)
