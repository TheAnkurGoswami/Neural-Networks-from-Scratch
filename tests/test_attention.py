import logging
from typing import List

import numpy as np
import pytest
import tensorflow as tf
import torch

from neural_networks.attention.scaled_dot_product_attention import ScaledDotProductAttention
from neural_networks.losses import CrossEntropyLoss, MSELoss, RMSELoss
from neural_networks.nn import Dense
from neural_networks.optimizers import get_optimizer
from tests.templates import (
    get_bias_template,
    get_loss_template,
    get_weight_template,
)
from utils import check_closeness

torch.set_printoptions(precision=10)
np.set_printoptions(precision=10)
tf.keras.backend.set_floatx("float32")
DEBUG = True



def test_self_attention():
    d_model = 3
    seq_len = 5
    batch_size = 1
    dim_kqv = 4

    x = np.random.randint(low=0, high=10, size=(batch_size, seq_len, d_model)).astype(np.float32)
    # Convert input to PyTorch tensor
    x_torch = torch.tensor(x, dtype=torch.float32)
    attention = ScaledDotProductAttention(d_model=d_model, dim_k=dim_kqv, dim_q=dim_kqv, dim_v=dim_kqv)

    attention.query_w, attention.key_w, attention.value_w

    output = attention.forward(x_torch.clone().detach())
    print("Output (Custom):", output)


    #  PYTORCH's self attention

    mha = torch.nn.MultiheadAttention(num_heads=1, embed_dim=d_model, batch_first=True)
    print(attention.query_w.shape)
    # Stack them vertically: QKV
    in_proj_weight = torch.cat([
        attention.query_w.clone().detach().requires_grad_(True),
        attention.key_w.clone().detach().requires_grad_(True),
        attention.value_w.clone().detach().requires_grad_(True)
    ], dim=0)  # [3*embed_dim, embed_dim]
    mha.in_proj_weight.data = in_proj_weight
    # mha.q_proj_weight.data = attention.query_w.clone().detach().requires_grad_(True)
    # mha.k_proj_weight.data = attention.key_w.clone().detach().requires_grad_(True)
    # mha.v_proj_weight.data = attention.value_w.clone().detach().requires_grad_(True)

    output = mha.forward(x_torch, x_torch, x_torch)
    print("Output (PT MHA):", output)

    # Convert weights to PyTorch tensors
    query_w_torch = attention.query_w.clone().detach().requires_grad_(True)
    key_w_torch = attention.key_w.clone().detach().requires_grad_(True)
    value_w_torch = attention.value_w.clone().detach().requires_grad_(True)

    # Compute queries, keys, and values
    queries = torch.matmul(x_torch, query_w_torch)
    keys = torch.matmul(x_torch, key_w_torch)
    values = torch.matmul(x_torch, value_w_torch)

    # Compute attention scores
    scores = torch.matmul(queries, keys.transpose(-2, -1)) / (dim_kqv ** 0.5)

    # Apply softmax to get attention weights
    attention_weights = torch.nn.functional.softmax(scores, dim=-1)

    # Compute the output
    output = torch.matmul(attention_weights, values)

    if DEBUG:
        # print("Queries (PT):", queries)
        # print("Keys (PT):", keys)
        # print("Values (PT):", values)
        print("Scores (PT):", scores)
        # print("Attention Weights (PT):", attention_weights)
        print("Output (PT):", output)

    # TensorFlow's self-attention
    # Convert input to TensorFlow tensor
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)

    # Compute queries, keys, and values
    queries_tf = tf.matmul(x_tf, tf.convert_to_tensor(attention.query_w, dtype=tf.float32))
    keys_tf = tf.matmul(x_tf, tf.convert_to_tensor(attention.key_w, dtype=tf.float32))
    values_tf = tf.matmul(x_tf, tf.convert_to_tensor(attention.value_w, dtype=tf.float32))

    # Compute attention scores
    scores_tf = tf.matmul(queries_tf, keys_tf, transpose_b=True) / tf.sqrt(tf.cast(dim_kqv, tf.float32))

    # Apply softmax to get attention weights
    attention_weights_tf = tf.nn.softmax(scores_tf, axis=-1)

    # Compute the output
    output_tf = tf.matmul(attention_weights_tf, values_tf)
    # torch._scaled_dot_product_attention_math()
    if DEBUG:
        # tf.print("Queries (TF):", queries_tf)
        # tf.print("Keys (TF):", keys_tf)
        # tf.print("Values (TF):", values_tf)
        tf.print("Scores (TF):", scores_tf)
        # tf.print("Attention Weights (TF):", attention_weights_tf)
        tf.print("Output (TF):", output_tf)
    assert False
    return output

