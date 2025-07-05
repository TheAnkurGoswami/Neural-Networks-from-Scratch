from collections import OrderedDict
from typing import Optional

from neural_networks.attention.projection import Projection
from neural_networks.attention.scaled_dot_product_attention import (
    ScaledDotProductAttention,
)
from neural_networks.backend import ARRAY_TYPE, get_backend
from neural_networks.constants import BACKPROP_WARN
from neural_networks.layers import Dense
from neural_networks.optimizers.base import Optimizer


class MultiHeadAttention:
    """
    Implements Multi-Head Attention mechanism.

    Multi-Head Attention allows the model to jointly attend to information
    from different representation subspaces at different positions.
    It consists of several parallel attention "heads". Each head performs
    Scaled Dot-Product Attention independently. The outputs of the heads are
    concatenated and linearly projected to produce the final output.

    Workflow:
    1. Input \(X\) is linearly projected to create Query (\(Q\)), Key (\(K\)),
        and Value (\(V\)) matrices for each head. These projections are
        typically done such that if \(d_{model}\) is the input dimension and
        \(h\) is the number of heads, each head operates on dimensions
        \(d_q/h, d_k/h, d_v/h\). Often \(d_q = d_k = d_v = d_{model}\).
        So, \(Q_i, K_i, V_i\) are for head \(i\).
    2. Each head computes attention:
        \( \text{head}_i = \text{Attention}(Q_i, K_i, V_i) \).
        The Attention is Scaled Dot-Product Attention.
    3. The outputs of all heads are concatenated:
        \( \text{Concat}(\text{head}_1, ..., \text{head}_h) \).
    4. The concatenated output is linearly projected by an output weight matrix
        \(W^O\).
       .. math::
           \text{MultiHead}(Q, K, V) =
           \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O

    Attributes:
        d_model (int): Dimensionality of the input and output features.
        num_heads (int): Number of parallel attention heads.
        dim_q (int): Total dimensionality for queries across all heads.
        dim_k (int): Total dimensionality for keys across all heads.
        dim_v (int): Total dimensionality for values across all heads.
        add_bias (bool): Whether to use bias in projection layers.
        proj_layer (OrderedDict[str, Projection]): Projection layers for
            Q, K, V.
        attn_heads (list[ScaledDotProductAttention]): List of attention heads.
        out_proj (Dense): Final linear projection layer.
        projections (dict): Stores the full Q, K, V projections during forward
            pass.
        output_head_size (int): Size of the output from a single attention
            head.
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
            d_model (int): Dimensionality of the input features
                (embedding size).
            num_heads (int): Number of attention heads.
            dim_q (Optional[int], optional): Total dimension for queries.
                Defaults to d_model.
            dim_k (Optional[int], optional): Total dimension for keys.
                Defaults to d_model.
            dim_v (Optional[int], optional): Total dimension for values.
                Defaults to d_model.
                Typically, d_model = num_heads * dim_v_head.
            add_bias (bool, optional): Whether to include bias terms in the
                projection layers. Defaults to True.
        """
        self.d_model: int = d_model  # Store as int, not tensor
        self.num_heads: int = num_heads
        self.add_bias: bool = add_bias

        # If specific dimensions for Q, K, V are not provided,
        # default them to d_model
        self.dim_q: int = dim_q if dim_q is not None else d_model
        self.dim_k: int = dim_k if dim_k is not None else d_model
        self.dim_v: int = dim_v if dim_v is not None else d_model

        if not (
            self.dim_q % num_heads == 0
            and self.dim_k % num_heads == 0
            and self.dim_v % num_heads == 0
        ):
            raise ValueError(
                "Total dimensions for Q, K, V must be divisible by num_heads."
            )

        # This map stores the total dimension for each projection type
        # (query, key, value)
        self.parameter_dims_map: OrderedDict[str, int] = OrderedDict(
            {
                "query": self.dim_q,
                "key": self.dim_k,
                "value": self.dim_v,
            }
        )

        self.proj_layer: OrderedDict[str, Projection] = OrderedDict()
        self.attn_heads: list[ScaledDotProductAttention] = []
        self.out_proj: Optional[Dense] = None  # Will be initialized in _build
        self.projections: dict = {}
        self.output_head_size: Optional[int] = None

        self._build()

    def _build(self) -> None:
        """
        Initializes the projection layers, attention heads, and the output
        projection layer.
        - Projection layers transform the input `d_model` to `dim_q_total`,
            `dim_k_total`, `dim_v_total`.
        - `num_heads` instances of `ScaledDotProductAttention` are created.
        - An output `Dense` layer projects the concatenated head outputs back
            to `d_model`.
        """
        for param_type, total_dim in self.parameter_dims_map.items():
            # Each projection layer takes d_model_val as input and outputs the
            # total_dim for Q, K, or V
            self.proj_layer[param_type] = Projection(
                in_features=self.d_model,
                out_features=total_dim,
                add_bias=self.add_bias,
            )

        self.attn_heads = [
            ScaledDotProductAttention() for _ in range(self.num_heads)
        ]

        # The input to out_proj is the concatenation of all head outputs.
        # Each head outputs dim_v_head. So, total
        # input features = num_heads * dim_v_head.
        # This should be equal to dim_v_total.
        # The output of out_proj is d_model.
        self.out_proj = Dense(
            in_features=self.dim_v,
            out_features=self.d_model,
            add_bias=self.add_bias,
        )

    def _get_head_projection(
        self, projection_name: str, head_ix: int
    ) -> ARRAY_TYPE:
        """
        Extracts the segment of the full Q, K, or V projection corresponding to
        a specific head.

        Args:
            projection_name (str): "query", "key", or "value".
            head_ix (int): Index of the head.
        Returns:
            ARRAY_TYPE: The segment for the specified head.
                Shape (batch_size, seq_len, dim_head_for_projection).
        """
        # Determine the dimension per head for this specific projection type
        match projection_name:
            case "query":
                head_dim = self.dim_q // self.num_heads
            case "key":
                head_dim = self.dim_k // self.num_heads
            case "value":
                head_dim = self.dim_v // self.num_heads
            case _:
                raise ValueError(f"Unknown projection name: {projection_name}")

        start_idx = head_ix * head_dim
        end_idx = start_idx + head_dim

        # Assuming full_projection has shape (batch_size, seq_len, total_dim)
        # We want to slice along the last dimension.
        return self.projections[projection_name][:, :, start_idx:end_idx]

    def forward(self, inputs: ARRAY_TYPE) -> ARRAY_TYPE:
        r"""
        Performs the forward pass of the Multi-Head Attention mechanism.

        Args:
            inputs (ARRAY_TYPE): Input tensor.
                Shape: (batch_size, seq_len, d_model).
                In self-attention, Q, K, V are derived from the same input.
                For encoder-decoder attention, K and V might come from encoder
                output. This implementation assumes self-attention for
                simplicity in projection. (If Q, K, V come from different
                sources, they should be passed separately)

        Returns:
            ARRAY_TYPE: Output tensor after multi-head attention and final
             projection. Shape: (batch_size, seq_len, d_model).
        """
        if self.out_proj is None:  # Should be initialized by _build
            raise RuntimeError("Output projection layer not initialized.")

        # 1. Linearly project inputs to get Q_full, K_full, V_full
        for param in self.parameter_dims_map.keys():
            self.projections[param] = self.proj_layer[param].forward(inputs)

        all_head_outputs = []
        for head_ix, head in enumerate(self.attn_heads):
            # 2. Pass Q_i, K_i, V_i to the i-th attention head
            output = head.forward(
                q_proj=self._get_head_projection("query", head_ix),
                k_proj=self._get_head_projection("key", head_ix),
                v_proj=self._get_head_projection("value", head_ix),
            )
            self.output_head_size = output.shape[-1]
            all_head_outputs.append(output)

        # 3. Concatenate outputs of all heads
        # Concatenating along the last dimension (feature dimension)
        # Resulting shape: (batch_size, seq_len, num_heads * dim_v_head)
        # which is (batch, seq, dim_v_total)
        concatenated_heads = get_backend()[0].cat(all_head_outputs, dim=-1)
        # Shape of concatenated_output: (batch_size, seq_len, d_model)
        return self.out_proj.forward(concatenated_heads)

    def backprop(
        self, dA: ARRAY_TYPE, optimizer: Optimizer
    ) -> None:  # Returns dX (gradient w.r.t input)
        r"""
        Performs the backward pass for Multi-Head Attention.

        Args:
            dA (ARRAY_TYPE): Gradient of the loss w.r.t the output of
                MultiHeadAttention. Shape: (batch_size, seq_len, d_model).
            optimizer (Optimizer): Optimizer instance for updating parameters.

        Returns:
            None: Gradients are propagated to internal projection layers.
                  (Technically, should return dInputs for consistency if this
                  layer is part of a sequence) For now, let's assume it updates
                  its own params and propagates to Q,K,V input projections.
        """
        if self.out_proj is None or self.output_head_size is None:
            raise ValueError(BACKPROP_WARN)
        # 1. Backprop through the final output projection (self.out_proj)
        # dA is dL/d(out_proj_output)
        # dOut is dL/d(out_proj_input) = dL/d(Concatenated_heads_output)
        # Shape: (batch_size, seq_len, num_heads * dim_v_head) or
        # (batch, seq, dim_v_total)
        dOut = self.out_proj.backprop(dA, optimizer)

        # Initialize lists to store gradients for full Q, K, V projections
        # These will accumulate gradients from all heads for each projection
        # type. Initialize with zeros to correctly accumulate.
        # Backend access needed for zeros_like.
        backend, _ = get_backend()
        # Using the shapes from the stored full projections
        all_dQ = []
        all_dK = []
        all_dV = []

        # 2. Backprop through concatenation and individual attention heads
        for head_ix, head in enumerate(self.attn_heads):
            # Slice dConcat_heads to get the gradient for the i-th head's
            # output. Each head outputs dim_v_head features.
            start, end = (
                self.output_head_size * head_ix,
                self.output_head_size * (head_ix + 1),
            )
            # Backprop through the i-th attention head
            # dQ_head_i, dK_head_i, dV_head_i are grads w.r.t. inputs of
            # ScaledDotProductAttention head i
            # Shapes: (batch, seq, dim_q_head), (batch, seq, dim_k_head),
            # (batch, seq, dim_v_head)
            dQ, dK, dV = head.backprop(dOut[:, :, start:end])
            # Accumulate these gradients into the corresponding slices.
            all_dQ.append(dQ)
            all_dK.append(dK)
            all_dV.append(dV)

        all_dQ = backend.cat(all_dQ, dim=-1)
        all_dK = backend.cat(all_dK, dim=-1)
        all_dV = backend.cat(all_dV, dim=-1)

        # 3. Backprop through the initial Q, K, V projection layers
        # all_dQ is dL/d(Q_full_projection_output)
        # all_dK is dL/d(K_full_projection_output)
        # all_dV is dL/d(V_full_projection_output)
        # These backprop calls will update the Wq, Wk, Wv projection matrices
        # and will return dL/d(input_to_projections), which is
        # dL/d(original_inputs_X)
        # Since Q,K,V projections might take same input X, their gradients
        # w.r.t X need to be summed.

        # Gradients w.r.t the input X from each projection path
        dX_q_proj = self.proj_layer["query"].backprop(all_dQ, optimizer)
        dX_k_proj = self.proj_layer["key"].backprop(all_dK, optimizer)
        dX_v_proj = self.proj_layer["value"].backprop(all_dV, optimizer)

        # Sum gradients from different paths if Q, K, V are derived from the
        # same input X. This is typical in self-attention.
        dX_total = dX_q_proj + dX_k_proj + dX_v_proj
        return dX_total
