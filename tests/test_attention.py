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
from utils import check_closeness, ScaledDotProductAttentionPytorch, ScaledDotProductAttentionTensorflow

torch.set_printoptions(precision=10)
np.set_printoptions(precision=10)
tf.keras.backend.set_floatx("float32")
DEBUG = True



def test_scaled_dot_product_attention():
    d_model = 3
    seq_len = 5
    batch_size = 1
    dim_kqv = d_model
    np.random.seed(100)
    torch.manual_seed(100)
    epochs = 1
    learning_rate = 0.001

    x = np.random.randint(low=0, high=10, size=(batch_size, seq_len, d_model)).astype(np.float32)
    y = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)

    # Convert input to PyTorch tensor
    x_torch = torch.tensor(x.astype(np.float32))
    y_torch = torch.tensor(y.astype(np.float32))
    # Convert input to TensorFlow tensor
    x_tf = tf.constant(x.astype(np.float32))
    y_tf = tf.constant(y.astype(np.float32))




    sdpa_cus = ScaledDotProductAttention(d_model=d_model, dim_k=dim_kqv, dim_v=dim_kqv)

    sdpa_pt = ScaledDotProductAttentionPytorch(
        d_model=d_model, dim_k=dim_kqv, dim_v=dim_kqv)
    sdpa_pt.set_weights(sdpa_cus.weights["query"].clone().detach().numpy(),
                        sdpa_cus.weights["key"].clone().detach().numpy(),
                        sdpa_cus.weights["value"].clone().detach().numpy())

    sdpa_tf = ScaledDotProductAttentionTensorflow(
        d_model=d_model, dim_k=dim_kqv, dim_v=dim_kqv)
    sdpa_tf.set_weights(sdpa_cus.weights["query"].clone().detach().numpy(),
                        sdpa_cus.weights["key"].clone().detach().numpy(),
                        sdpa_cus.weights["value"].clone().detach().numpy())

    # Initialize loss functions and optimizers for each framework
    loss = RMSELoss()
    loss_torch = torch.nn.MSELoss()
    loss_tf = tf.keras.losses.MeanSquaredError()

    optimizer = get_optimizer("adam")(learning_rate=learning_rate)
    optimizer_tf = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer_torch = torch.optim.Adam(
        params=[sdpa_pt.W_key, sdpa_pt.W_query, sdpa_pt.W_value],
        lr=learning_rate
    )

    output_cus = sdpa_cus.forward(x_torch.clone().detach())
    cost_cus = loss.forward(output_cus, y_torch)

    output_pt = sdpa_pt.forward(x_torch)
    cost_pt = torch.sqrt(loss_torch(output_pt, y_torch))

    with tf.GradientTape() as tape:
        output_tf = sdpa_tf.forward(x_tf)
        cost_tf = tf.sqrt(loss_tf(output_tf, y_tf))

    # Backward pass and optimization
    dL = loss.backprop()
    # print(dL)
    optimizer.set_cur_epoch(epochs + 1)
    sdpa_cus.backprop(dL, optimizer)

    cost_pt.backward()
    optimizer_torch.step()
    print("PT Gradient for attn scores", sdpa_pt.scores.grad)
    print("PT Gradient for attn weights", sdpa_pt.attention_weights.grad)
    for param in [sdpa_pt.W_query, sdpa_pt.W_key, sdpa_pt.W_value]:
        print(f"PT Gradient for {param.grad}")
    optimizer_torch.zero_grad()

    trainable_variables = [sdpa_tf.W_query, sdpa_tf.W_key, sdpa_tf.W_value]
    grads = tape.gradient(
        cost_tf, trainable_variables)
    # if DEBUG:
    #     print("Grads (TF):", grads)
    optimizer_tf.apply_gradients(
        zip(grads, trainable_variables))

    # print("Loss (Custom):", cost_cus)
    # print("Loss (PT):", cost_pt.item())
    # print("Loss (TF):", cost_tf.numpy())
    assert check_closeness(
        cost_cus.detach().numpy(), cost_tf
    ), f"{get_loss_template('tf')}"
    assert check_closeness(
        cost_cus.detach().numpy(), cost_pt.item()
    ), f"{get_loss_template('pt')}"


    # Check closeness of outputs
    assert check_closeness(
        output_cus.detach().numpy(), output_tf.numpy()
    ), f"{get_weight_template('tf')}"
    assert check_closeness(
        output_cus.detach().numpy(), output_pt.detach().numpy()
    ), f"{get_weight_template('pt')}"

    # Check closeness of weights
    for key, W_tf, W_pt in zip(
        ["query", "key", "value"],
        [sdpa_tf.W_query, sdpa_tf.W_key, sdpa_tf.W_value],
        [sdpa_pt.W_query, sdpa_pt.W_key, sdpa_pt.W_value],
    ):
        # if key in ["query", "key"]:
        #     continue
        # print(
        #     sdpa_cus.weights[key].detach().numpy(),
        #     W_tf.numpy(),
        #     W_pt.detach().numpy(),
        #     sep="\n")
        assert check_closeness(
            sdpa_cus.weights[key].detach().numpy(), W_tf.numpy()
        ), f"{get_weight_template('tf')}"
        assert check_closeness(
            sdpa_cus.weights[key].detach().numpy(), W_pt.detach().numpy()
        ), f"{get_weight_template('pt')}"

    assert False

