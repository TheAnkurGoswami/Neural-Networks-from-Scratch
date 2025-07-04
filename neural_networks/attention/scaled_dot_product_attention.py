from typing import Dict

import numpy as np
import torch as pt

from neural_networks.activations import Softmax
from neural_networks.attention.projection import Projection
from neural_networks.backend import ARRAY_TYPE, get_backend


class ScaledDotProductAttention:
    """
    Implements the Scaled Dot-Product Attention mechanism.

    This is a core component of Transformer models, allowing the model to weigh the
    importance of different parts of the input sequence when processing information.
    The attention score is calculated as:
    Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) ) V
    where Q, K, V are Query, Key, and Value projections respectively, and d_k is the
    dimension of the Key vectors.

    Attributes:
        proj_layer (Dict[str, Projection]): Not directly used by this class if Q, K, V are pre-computed.
                                            Retained for potential future use or context.
        dim_k (Optional[ARRAY_TYPE]): Dimension of the key vectors, stored during forward pass.
        projections (Dict[str, ARRAY_TYPE]): Stores Q, K, V projections during forward pass.
        softmax (Softmax): Instance of the Softmax activation function.
        softmax_attn (ARRAY_TYPE): Attention weights after applying softmax, stored during forward pass.
        attention (ARRAY_TYPE): Output of the attention mechanism, stored during forward pass.
    """

    # proj_layer: Dict[str, Projection] = {} # This seems to be a class attribute, might be better as instance or not needed if Q,K,V are passed in.
                                         # For now, will assume Q, K, V are already projected inputs.

    def __init__(self, softmax_do_clip: bool = False):
        """
        Initializes the ScaledDotProductAttention layer.

        Args:
            softmax_do_clip (bool, optional): Whether the Softmax function used internally
                                              should clip its output. Defaults to False,
                                              as clipping is often not needed before multiplication
                                              with Value vectors if CrossEntropyLoss handles logits.
        """
        self.dim_k: Optional[ARRAY_TYPE] = None
        self.projections: Dict[str, ARRAY_TYPE] = {}
        # Softmax is applied on attention scores. Clipping is usually not needed here
        # unless the output is directly fed into a log function elsewhere.
        self.softmax = Softmax(do_clip=softmax_do_clip)
        self.softmax_attn: Optional[ARRAY_TYPE] = None
        self.attention: Optional[ARRAY_TYPE] = None


    def forward(self, q_proj: ARRAY_TYPE, k_proj: ARRAY_TYPE, v_proj: ARRAY_TYPE) -> ARRAY_TYPE:
        r"""
        Performs the forward pass of the Scaled Dot-Product Attention.

        The formula is:
        .. math::
            \text{scores} = Q K^T \\
            \text{scaled\_scores} = \frac{\text{scores}}{\sqrt{d_k}} \\
            \text{attention\_weights} = \text{softmax}(\text{scaled\_scores}) \\
            \text{output} = \text{attention\_weights} V

        Args:
            q_proj (ARRAY_TYPE): Query projection. Shape: (batch_size, seq_len_q, dim_q)
            k_proj (ARRAY_TYPE): Key projection. Shape: (batch_size, seq_len_kv, dim_k)
            v_proj (ARRAY_TYPE): Value projection. Shape: (batch_size, seq_len_kv, dim_v)
                                 Note: dim_q should be equal to dim_k for dot product.
                                 seq_len_q is the length of the query sequence.
                                 seq_len_kv is the length of the key/value sequence.

        Returns:
            ARRAY_TYPE: The attention output. Shape: (batch_size, seq_len_q, dim_v)
        """
        self.projections = {
            "query": q_proj,
            "key": k_proj,
            "value": v_proj,
        }

        backend, backend_module = get_backend()

        # Q K^T
        # Q: (batch, seq_len_q, dim_q)
        # K.T: (batch, dim_k, seq_len_kv)
        # Assuming dim_q == dim_k
        # Result (attention_scores): (batch, seq_len_q, seq_len_kv)
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

    def backprop(self, dA: ARRAY_TYPE) -> Tuple[ARRAY_TYPE, ARRAY_TYPE, ARRAY_TYPE]:
        r"""
        Performs the backward pass for Scaled Dot-Product Attention.

        Given \( dA = \frac{\partial L}{\partial A} \) (gradient of loss w.r.t. attention output),
        this computes \( dQ_{proj}, dK_{proj}, dV_{proj} \) (gradients w.r.t. Q, K, V inputs to this block).

        Let \( A = \text{softmax}(S) V \), where \( S = \frac{QK^T}{\sqrt{d_k}} \).
        The gradients are:
        1.  Gradient w.r.t. \(V\) (Value):
            .. math::
                dV_{proj} = \text{softmax}(S)^T dA

        2.  Gradient w.r.t. softmax output (\( \text{softmax}(S) \)):
            .. math::
                d\text{softmax}(S) = dA V^T

        3.  Gradient w.r.t. scaled scores (\(S\)) (passing through softmax.backprop):
            Let \( dS' = d\text{softmax}(S) \).
            .. math::
                dS = \text{softmax.backprop}(dS')
                   = \text{softmax}(S) \odot (dS' - \text{sum}(dS' \odot \text{softmax}(S)))

        4.  Gradient w.r.t. unscaled scores (undo scaling):
            .. math::
                d(QK^T) = \frac{dS}{\sqrt{d_k}}

        5.  Gradient w.r.t. \(Q_{proj}\) (Query):
            .. math::
                dQ_{proj} = d(QK^T) K

        6.  Gradient w.r.t. \(K_{proj}\) (Key):
            .. math::
                dK_{proj} = d(QK^T)^T Q


        Args:
            dA (ARRAY_TYPE): Gradient of the loss with respect to the attention output.
                             Shape: (batch_size, seq_len_q, dim_v).

        Returns:
            Tuple[ARRAY_TYPE, ARRAY_TYPE, ARRAY_TYPE]:
                - \( dQ_{proj} \): Gradient w.r.t. Query projection. Shape: (batch_size, seq_len_q, dim_q)
                - \( dK_{proj} \): Gradient w.r.t. Key projection. Shape: (batch_size, seq_len_kv, dim_k)
                - \( dV_{proj} \): Gradient w.r.t. Value projection. Shape: (batch_size, seq_len_kv, dim_v)
        """
        if self.softmax_attn is None or \
           self.projections.get("value") is None or \
           self.projections.get("key") is None or \
           self.projections.get("query") is None or \
           self.dim_k is None:
            raise ValueError("Forward pass must be called before backprop, or attributes not set.")

        backend, _ = get_backend()

        # dA Shape: (batch, seq_len_q, dim_v)
        # self.softmax_attn (attention_weights): (batch, seq_len_q, seq_len_kv)
        # self.projections["value"] (V): (batch, seq_len_kv, dim_v)
        # self.projections["key"] (K): (batch, seq_len_kv, dim_k)
        # self.projections["query"] (Q): (batch, seq_len_q, dim_q)

        # 1. Calculate dV_proj = softmax_attn^T @ dA
        # softmax_attn^T: (batch, seq_len_kv, seq_len_q)
        # dA: (batch, seq_len_q, dim_v)
        # dV_proj: (batch, seq_len_kv, dim_v)
        dV_proj = backend.matmul(self.softmax_attn.transpose(-1, -2), dA)

        # 2. Calculate dSoftmaxAttn = dA @ V^T
        # dA: (batch, seq_len_q, dim_v)
        # V^T: (batch, dim_v, seq_len_kv)
        # dSoftmaxAttn: (batch, seq_len_q, seq_len_kv)
        dSoftmaxAttn = backend.matmul(dA, self.projections["value"].transpose(-1, -2))

        # 3. Backprop through softmax: dScaledScores = self.softmax.backprop(dSoftmaxAttn)
        # Need to reshape for softmax backprop if it expects 2D input (batch*seq_len_q, seq_len_kv)
        batch_dim, seq_len_q_dim, seq_len_kv_dim = dSoftmaxAttn.shape
        dSoftmaxAttn_reshaped = dSoftmaxAttn.reshape((batch_dim * seq_len_q_dim, seq_len_kv_dim))

        # Softmax backprop expects dL/d(softmax_output)
        # self.softmax internal state (_activation) was set with reshaped input during forward.
        # So, dSoftmaxAttn_reshaped is appropriate here.
        dScaledScores_reshaped = self.softmax.backprop(dSoftmaxAttn_reshaped)
        dScaledScores = dScaledScores_reshaped.reshape((batch_dim, seq_len_q_dim, seq_len_kv_dim))

        # 4. Undo scaling: dUnscaledScores = dScaledScores / sqrt(d_k)
        dUnscaledScores = dScaledScores / backend.sqrt(self.dim_k) # d(QK^T)

        # 5. Calculate dQ_proj = dUnscaledScores @ K
        # dUnscaledScores (d(QK^T)): (batch, seq_len_q, seq_len_kv)
        # K: (batch, seq_len_kv, dim_k) (dim_k is also dim_q here)
        # dQ_proj: (batch, seq_len_q, dim_k) which is dim_q
        dQ_proj = backend.matmul(dUnscaledScores, self.projections["key"])

        # 6. Calculate dK_proj = dUnscaledScores^T @ Q
        # dUnscaledScores^T: (batch, seq_len_kv, seq_len_q)
        # Q: (batch, seq_len_q, dim_q) (dim_q is also dim_k here)
        # dK_proj: (batch, seq_len_kv, dim_q) which is dim_k
        dK_proj = backend.matmul(dUnscaledScores.transpose(-1, -2), self.projections["query"])

        return dQ_proj, dK_proj, dV_proj
