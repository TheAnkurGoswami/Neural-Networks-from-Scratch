from collections import OrderedDict
from typing import Optional

import numpy as np
import torch as pt

from neural_networks.attention.projection import Projection
from neural_networks.attention.scaled_dot_product_attention import (
    ScaledDotProductAttention,
)
from neural_networks.backend import ARRAY_TYPE, get_backend
from neural_networks.layers import Dense # Updated: Import Dense from layers

# Alias for compatibility with HuggingFace's BERT implementation
# MultiheadAttention is imported above as BertSelfAttention # This comment seems outdated.


class MultiHeadAttention:
    """
    Implements Multi-Head Attention mechanism.

    Multi-Head Attention allows the model to jointly attend to information
    from different representation subspaces at different positions.
    It consists of several parallel attention "heads". Each head performs
    Scaled Dot-Product Attention independently. The outputs of the heads are
    concatenated and linearly projected to produce the final output.

    Workflow:
    1. Input \(X\) is linearly projected to create Query (\(Q\)), Key (\(K\)), and Value (\(V\)) matrices
       for each head. These projections are typically done such that if \(d_{model}\) is the input dimension
       and \(h\) is the number of heads, each head operates on dimensions \(d_q/h, d_k/h, d_v/h\).
       Often \(d_q = d_k = d_v = d_{model}\).
       So, \(Q_i, K_i, V_i\) are for head \(i\).
    2. Each head computes attention: \( \text{head}_i = \text{Attention}(Q_i, K_i, V_i) \).
       The Attention is Scaled Dot-Product Attention.
    3. The outputs of all heads are concatenated: \( \text{Concat}(\text{head}_1, ..., \text{head}_h) \).
    4. The concatenated output is linearly projected by an output weight matrix \(W^O\).
       .. math::
           \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O

    Attributes:
        d_model (int): Dimensionality of the input and output features.
        num_heads (int): Number of parallel attention heads.
        dim_q (int): Total dimensionality for queries across all heads.
        dim_k (int): Total dimensionality for keys across all heads.
        dim_v (int): Total dimensionality for values across all heads.
        add_bias (bool): Whether to use bias in projection layers.
        dim_q_head (int): Dimensionality of query per head.
        dim_k_head (int): Dimensionality of key per head.
        dim_v_head (int): Dimensionality of value per head.
        proj_layer (OrderedDict[str, Projection]): Projection layers for Q, K, V.
        attn_heads (list[ScaledDotProductAttention]): List of attention heads.
        out_proj (Dense): Final linear projection layer.
        projections (dict): Stores the full Q, K, V projections during forward pass.
        output_head_size (int): Size of the output from a single attention head.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        # If dim_q, dim_k, dim_v are not specified, they default to d_model
        dim_q: Optional[int] = None,
        dim_k: Optional[int] = None,
        dim_v: Optional[int] = None,
        add_bias: bool = True,
        # TODO: Add dropout option for attention weights and output projection
    ) -> None:
        """
        Initializes the MultiHeadAttention layer.

        Args:
            d_model (int): Dimensionality of the input features (embedding size).
            num_heads (int): Number of attention heads.
            dim_q (Optional[int], optional): Total dimension for queries. Defaults to d_model.
            dim_k (Optional[int], optional): Total dimension for keys. Defaults to d_model.
            dim_v (Optional[int], optional): Total dimension for values. Defaults to d_model.
                                          Typically, d_model = num_heads * dim_v_head.
            add_bias (bool, optional): Whether to include bias terms in the projection layers.
                                       Defaults to True.
        """
        # self.output_w: Optional[ARRAY_TYPE] = None # This variable is not used.
        self.d_model_val: int = d_model # Store as int, not tensor
        self.num_heads: int = num_heads
        self.add_bias: bool = add_bias

        # If specific dimensions for Q, K, V are not provided, default them to d_model
        self.dim_q_total: int = dim_q if dim_q is not None else d_model
        self.dim_k_total: int = dim_k if dim_k is not None else d_model
        self.dim_v_total: int = dim_v if dim_v is not None else d_model

        if not (self.dim_q_total % num_heads == 0 and \
                self.dim_k_total % num_heads == 0 and \
                self.dim_v_total % num_heads == 0):
            raise ValueError("Total dimensions for Q, K, V must be divisible by num_heads.")

        self.dim_q_head: int = self.dim_q_total // num_heads
        self.dim_k_head: int = self.dim_k_total // num_heads
        self.dim_v_head: int = self.dim_v_total // num_heads

        # This map stores the total dimension for each projection type (query, key, value)
        self.parameter_dims_map: OrderedDict[str, int] = OrderedDict(
            {
                "query": self.dim_q_total,
                "key": self.dim_k_total,
                "value": self.dim_v_total,
            }
        )

        self.proj_layer: OrderedDict[str, Projection] = OrderedDict()
        self.attn_heads: list[ScaledDotProductAttention] = []
        self.out_proj: Optional[Dense] = None # Will be initialized in _build
        self.projections: dict = {} # To store Q, K, V after projection in forward pass
        self.output_head_size: Optional[int] = None # To store size of individual head output

        self._build()

    def _build(self) -> None:
        """
        Initializes the projection layers, attention heads, and the output projection layer.
        - Projection layers transform the input `d_model` to `dim_q_total`, `dim_k_total`, `dim_v_total`.
        - `num_heads` instances of `ScaledDotProductAttention` are created.
        - An output `Dense` layer projects the concatenated head outputs back to `d_model`.
        """
        for param_type, total_dim in self.parameter_dims_map.items():
            # Each projection layer takes d_model_val as input and outputs the total_dim for Q, K, or V
            self.proj_layer[param_type] = Projection(
                in_features=self.d_model_val, # Input to Q,K,V projections is d_model
                out_features=total_dim,      # Output is the total dim for that projection
                add_bias=self.add_bias,
                activation=None # Projections are linear
            )

        self.attn_heads = [
            ScaledDotProductAttention() for _ in range(self.num_heads)
        ]

        # The input to out_proj is the concatenation of all head outputs.
        # Each head outputs dim_v_head. So, total input features = num_heads * dim_v_head.
        # This should be equal to dim_v_total.
        # The output of out_proj is d_model.
        self.out_proj = Dense(
            in_features=self.num_heads * self.dim_v_head, # This is effectively self.dim_v_total
            out_features=self.d_model_val,
            activation=None, # Output projection is typically linear
            add_bias=self.add_bias
        )

    def _get_head_projection(self, projection_name: str, head_ix: int, full_projection: ARRAY_TYPE) -> ARRAY_TYPE:
        """
        Extracts the segment of the full Q, K, or V projection corresponding to a specific head.

        Args:
            projection_name (str): "query", "key", or "value".
            head_ix (int): Index of the head.
            full_projection (ARRAY_TYPE): The complete Q, K, or V projection tensor
                                          of shape (batch_size, seq_len, total_dim_for_projection).
        Returns:
            ARRAY_TYPE: The segment for the specified head.
                        Shape (batch_size, seq_len, dim_head_for_projection).
        """
        # Determine the dimension per head for this specific projection type
        if projection_name == "query":
            head_dim = self.dim_q_head
        elif projection_name == "key":
            head_dim = self.dim_k_head
        elif projection_name == "value":
            head_dim = self.dim_v_head
        else:
            raise ValueError(f"Unknown projection name: {projection_name}")

        start_idx = head_ix * head_dim
        end_idx = start_idx + head_dim

        # Assuming full_projection has shape (batch_size, seq_len, total_dim)
        # We want to slice along the last dimension.
        return full_projection[:, :, start_idx:end_idx]

    def forward(self, inputs: ARRAY_TYPE) -> ARRAY_TYPE:
        r"""
        Performs the forward pass of the Multi-Head Attention mechanism.

        Args:
            inputs (ARRAY_TYPE): Input tensor. Shape: (batch_size, seq_len, d_model).
                                 In self-attention, Q, K, V are derived from the same input.
                                 For encoder-decoder attention, K and V might come from encoder output.
                                 This implementation assumes self-attention for simplicity in projection.
                                 (If Q, K, V come from different sources, they should be passed separately)

        Returns:
            ARRAY_TYPE: Output tensor after multi-head attention and final projection.
                        Shape: (batch_size, seq_len, d_model).
        """
        if self.out_proj is None: # Should be initialized by _build
            raise RuntimeError("Output projection layer not initialized.")

        # 1. Linearly project inputs to get Q_full, K_full, V_full
        # These projections map from d_model to dim_q_total, dim_k_total, dim_v_total respectively.
        self.projections["query"] = self.proj_layer["query"].forward(inputs) # (batch, seq, dim_q_total)
        self.projections["key"] = self.proj_layer["key"].forward(inputs)   # (batch, seq, dim_k_total)
        self.projections["value"] = self.proj_layer["value"].forward(inputs) # (batch, seq, dim_v_total)

        all_head_outputs = []
        for i in range(self.num_heads):
            # Get Q_i, K_i, V_i for the i-th head by slicing the full projections
            q_head_i = self._get_head_projection("query", i, self.projections["query"]) # (batch, seq, dim_q_head)
            k_head_i = self._get_head_projection("key", i, self.projections["key"])   # (batch, seq, dim_k_head)
            v_head_i = self._get_head_projection("value", i, self.projections["value"]) # (batch, seq, dim_v_head)

            # 2. Pass Q_i, K_i, V_i to the i-th attention head
            head_output_i = self.attn_heads[i].forward(q_head_i, k_head_i, v_head_i) # (batch, seq, dim_v_head)
            all_head_outputs.append(head_output_i)

            if i == 0 and head_output_i is not None: # Store for backprop logic if needed
                 self.output_head_size = head_output_i.shape[-1]


        # 3. Concatenate outputs of all heads
        # Concatenating along the last dimension (feature dimension)
        # Resulting shape: (batch_size, seq_len, num_heads * dim_v_head) which is (batch, seq, dim_v_total)
        concatenated_heads = get_backend()[0].cat(all_head_outputs, dim=-1)

        # 4. Pass concatenated output through the final linear projection layer
        # Input: (batch, seq, dim_v_total), Output: (batch, seq, d_model)
        multihead_output = self.out_proj.forward(concatenated_heads)

        return multihead_output

    def backprop(self, dA_multihead: ARRAY_TYPE, optimizer: Optimizer) -> None: # Returns dX (gradient w.r.t input)
        r"""
        Performs the backward pass for Multi-Head Attention.

        Args:
            dA_multihead (ARRAY_TYPE): Gradient of the loss with respect to the output of MultiHeadAttention.
                                       Shape: (batch_size, seq_len, d_model).
            optimizer (Optimizer): Optimizer instance for updating parameters.

        Returns:
            None: Gradients are propagated to internal projection layers.
                  (Technically, should return dInputs for consistency if this layer is part of a sequence)
                  For now, let's assume it updates its own params and propagates to Q,K,V input projections.
        """
        if self.out_proj is None or self.output_head_size is None:
             raise RuntimeError("Forward pass must be completed before backprop, or layers not initialized.")

        # 1. Backprop through the final output projection (self.out_proj)
        # dA_multihead is dL/d(out_proj_output)
        # dConcat_heads is dL/d(out_proj_input) = dL/d(Concatenated_heads_output)
        # Shape: (batch_size, seq_len, num_heads * dim_v_head) or (batch, seq, dim_v_total)
        dConcat_heads = self.out_proj.backprop(dA_multihead, optimizer)

        # Initialize lists to store gradients for full Q, K, V projections
        # These will accumulate gradients from all heads for each projection type.
        # Initialize with zeros to correctly accumulate.
        # Backend access needed for zeros_like.
        backend, _ = get_backend()
        # Using the shapes from the stored full projections
        dQuery_full = backend.zeros_like(self.projections["query"])
        dKey_full = backend.zeros_like(self.projections["key"])
        dValue_full = backend.zeros_like(self.projections["value"])

        # 2. Backprop through concatenation and individual attention heads
        for i in range(self.num_heads):
            # Slice dConcat_heads to get the gradient for the i-th head's output
            # Each head outputs dim_v_head features.
            # The original code used self.output_head_size, which should be self.dim_v_head.
            start_idx = i * self.dim_v_head
            end_idx = start_idx + self.dim_v_head

            dA_head_i = dConcat_heads[:, :, start_idx:end_idx] # Grad for output of head_i

            # Backprop through the i-th attention head
            # dQ_head_i, dK_head_i, dV_head_i are grads w.r.t. inputs of ScaledDotProductAttention head i
            # Shapes: (batch, seq, dim_q_head), (batch, seq, dim_k_head), (batch, seq, dim_v_head)
            dQ_head_i, dK_head_i, dV_head_i = self.attn_heads[i].backprop(dA_head_i)

            # Accumulate these gradients into the corresponding slices of dQuery_full, dKey_full, dValue_full
            # This is effectively reversing the _get_head_projection slicing.
            q_start_idx = i * self.dim_q_head; q_end_idx = q_start_idx + self.dim_q_head
            k_start_idx = i * self.dim_k_head; k_end_idx = k_start_idx + self.dim_k_head
            v_start_idx = i * self.dim_v_head; v_end_idx = v_start_idx + self.dim_v_head

            # This requires a backend-specific way to add to a slice.
            # For simplicity, assuming direct slice assignment if backend supports it,
            # or element-wise addition if slices are copied first.
            # Example for PyTorch: dQuery_full[:, :, q_start_idx:q_end_idx] += dQ_head_i
            # For NumPy: dQuery_full[:, :, q_start_idx:q_end_idx] = dQuery_full[:, :, q_start_idx:q_end_idx] + dQ_head_i
            # Generic way:
            current_q_slice = dQuery_full[:, :, q_start_idx:q_end_idx]
            dQuery_full[:, :, q_start_idx:q_end_idx] = current_q_slice + dQ_head_i

            current_k_slice = dKey_full[:, :, k_start_idx:k_end_idx]
            dKey_full[:, :, k_start_idx:k_end_idx] = current_k_slice + dK_head_i

            current_v_slice = dValue_full[:, :, v_start_idx:v_end_idx]
            dValue_full[:, :, v_start_idx:v_end_idx] = current_v_slice + dV_head_i

        # 3. Backprop through the initial Q, K, V projection layers
        # dQuery_full is dL/d(Q_full_projection_output)
        # dKey_full is dL/d(K_full_projection_output)
        # dValue_full is dL/d(V_full_projection_output)
        # These backprop calls will update the Wq, Wk, Wv projection matrices
        # and will return dL/d(input_to_projections), which is dL/d(original_inputs_X)
        # Since Q,K,V projections might take same input X, their gradients w.r.t X need to be summed.

        # Gradients w.r.t the input X from each projection path
        dX_from_q_proj = self.proj_layer["query"].backprop(dQuery_full, optimizer)
        dX_from_k_proj = self.proj_layer["key"].backprop(dKey_full, optimizer)
        dX_from_v_proj = self.proj_layer["value"].backprop(dValue_full, optimizer)

        # Sum gradients from different paths if Q, K, V are derived from the same input X
        # This is typical in self-attention.
        # dX_total = dX_from_q_proj + dX_from_k_proj + dX_from_v_proj
        # return dX_total
        # For now, the original code didn't return dX, just updated params.
        # Let's stick to that for now, but ideally, it should return the gradient w.r.t its input(s).
        # The current function signature is `-> None`.
        # If this layer were to be used sequentially, it would need to return dL/dX.
        # For now, we assume the optimizer handles all parameters updated within.
        # The original code also didn't sum dX from different projection layers.
        # Each proj_layer.backprop updates its own weights and returns dL/dX_proj_input.
        # If the inputs to these proj_layers were different, this would be fine.
        # If they are the same (as in self-attention from a single 'inputs'), then the dX returned
        # by each should be summed to get the total dL/dInputs.
        # The current code structure implies that the backprop for proj_layer handles its own updates.
        # The FIXME in the original __init__ about necessary parameters is relevant here.
        # If inputs for Q, K, V are different, this structure is mostly fine.
        # If same, then dL/dX needs to be aggregated.
        pass # No explicit return of dL/dX as per original structure's implication.
