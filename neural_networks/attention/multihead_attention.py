from collections import OrderedDict
from typing import Optional

import numpy as np
import torch as pt

from neural_networks.attention.projection import Projection
from neural_networks.attention.scaled_dot_product_attention import (
    ScaledDotProductAttention,
)
from neural_networks.backend import ARRAY_TYPE, get_backend
from neural_networks.nn import Dense

# Alias for compatibility with HuggingFace's BERT implementation
# MultiheadAttention is imported above as BertSelfAttention


class MultiHeadAttention:
    def __init__(
        # FIXME: Check if all parameters are necessary
        self,
        d_model: int,
        num_heads: int,
        dim_q: int,
        dim_k: int,
        dim_v: int,
        add_bias: bool = True,
    ) -> None:
        self.output_w: Optional[ARRAY_TYPE] = None
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        _, backend_module = get_backend()
        self.add_bias = add_bias

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

        assert (
            self.d_model % self.num_heads == 0
        ), "d_model must be divisible by num_heads"
        self.dim_q_head = self.d_model // self.num_heads
        self.dim_k_head = self.dim_k // self.num_heads
        self.dim_v_head = self.dim_v // self.num_heads

        self.parameter_dims_map: OrderedDict[str, int] = OrderedDict(
            {
                "query": dim_q if hasattr(self, "dim_q") else d_model,
                "key": dim_k,
                "value": dim_v,
            }
        )

        self.proj_layer: OrderedDict[str, Projection] = OrderedDict()
        self._build()

    def _build(self):
        """
        Initializes the weights for the multi-head attention mechanism.
        """

        for param, dim in self.parameter_dims_map.items():
            self.proj_layer[param] = Projection(
                in_features=self.d_model,
                out_features=dim,
                add_bias=self.add_bias,
            )
        self.attn_heads = [
            ScaledDotProductAttention() for _ in range(self.num_heads)
        ]
        self.out_proj = Dense(
            in_features=self.num_heads * self.dim_v_head,  # d_model
            out_features=self.d_model,
        )

    def get_head_projection(self, projection, head_ix):
        head_size = self.parameter_dims_map[projection] // self.num_heads
        start, end = (head_size * head_ix, head_size * (head_ix + 1))
        return self.projections[projection][:, :, start:end]

    def forward(self, inputs):
        """
        Forward pass of the multi-head attention mechanism.
        """

        self.projections = {}
        for param in self.parameter_dims_map.keys():
            self.projections[param] = self.proj_layer[param].forward(inputs)
        all_outputs = []
        for head_ix, head in enumerate(self.attn_heads):
            output = head.forward(
                q_proj=self.get_head_projection("query", head_ix),
                k_proj=self.get_head_projection("key", head_ix),
                v_proj=self.get_head_projection("value", head_ix),
            )
            self.output_head_size = output.shape[-1]
            all_outputs.append(output)
        concatenated_output = pt.cat(all_outputs, dim=-1)
        # Shape of concatenated_output: (batch_size, seq_len, d_model)
        return self.out_proj.forward(concatenated_output)

    def backprop(self, dA, optimizer):
        dOut = self.out_proj.backprop(dA, optimizer)
        all_dQ = []
        all_dK = []
        all_dV = []
        for head_ix, head in enumerate(self.attn_heads):
            start, end = (
                self.output_head_size * head_ix,
                self.output_head_size * (head_ix + 1),
            )
            dQ, dK, dV = head.backprop(dOut[:, :, start:end])
            all_dQ.append(dQ)
            all_dK.append(dK)
            all_dV.append(dV)
        all_dQ = pt.cat(all_dQ, dim=-1)
        all_dK = pt.cat(all_dK, dim=-1)
        all_dV = pt.cat(all_dV, dim=-1)

        self.proj_layer["query"].backprop(all_dQ, optimizer)
        self.proj_layer["key"].backprop(all_dK, optimizer)
        self.proj_layer["value"].backprop(all_dV, optimizer)
