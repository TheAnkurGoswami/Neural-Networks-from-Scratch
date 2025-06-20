from collections import defaultdict

import numpy as np
import pytest
import tensorflow as tf
import torch

from neural_networks.attention.multihead_attention import MultiHeadAttention
from neural_networks.attention.scaled_dot_product_attention import (
    ScaledDotProductAttention,
)
from neural_networks.losses import RMSELoss
from neural_networks.optimizers import get_optimizer
from tests.templates import (
    get_bias_template,
    get_loss_template,
    get_output_template,
    get_weight_template,
)
from utils import (
    ScaledDotProductAttentionPytorch,
    ScaledDotProductAttentionTensorflow,
    check_closeness,
)

torch.set_printoptions(precision=10)
np.set_printoptions(precision=10)
tf.keras.backend.set_floatx("float32")
DEBUG = True


# @pytest.mark.parametrize("d_model", [3, 5, 10])
# @pytest.mark.parametrize("batch_size", [1, 2, 64])
# @pytest.mark.parametrize("dim_kqv", [3, 7])
@pytest.mark.parametrize("add_bias", [True, False])
def test_scaled_dot_product_attention(add_bias: bool):
    d_model = 3
    seq_len = 5
    batch_size = 1
    dim_kqv = d_model
    np.random.seed(100)
    torch.manual_seed(100)
    learning_rate = 0.001

    x = np.random.randint(
        low=0, high=10, size=(batch_size, seq_len, d_model)
    ).astype(np.float32)
    y = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)

    # Convert input to PyTorch tensor
    x_torch = torch.tensor(x.astype(np.float32))
    y_torch = torch.tensor(y.astype(np.float32))
    # Convert input to TensorFlow tensor
    x_tf = tf.constant(x.astype(np.float32))
    y_tf = tf.constant(y.astype(np.float32))

    sdpa_cus = ScaledDotProductAttention(
        d_model=d_model, dim_k=dim_kqv, dim_v=dim_kqv, add_bias=add_bias
    )

    sdpa_pt = ScaledDotProductAttentionPytorch(
        d_model=d_model, dim_k=dim_kqv, dim_v=dim_kqv, add_bias=add_bias
    )
    sdpa_pt.set_weights(
        sdpa_cus.proj_layer["query"]._weights.clone().detach().numpy(),
        sdpa_cus.proj_layer["key"]._weights.clone().detach().numpy(),
        sdpa_cus.proj_layer["value"]._weights.clone().detach().numpy(),
    )

    sdpa_tf = ScaledDotProductAttentionTensorflow(
        d_model=d_model, dim_k=dim_kqv, dim_v=dim_kqv, add_bias=add_bias
    )
    sdpa_tf.set_weights(
        sdpa_cus.proj_layer["query"]._weights.clone().detach().numpy(),
        sdpa_cus.proj_layer["key"]._weights.clone().detach().numpy(),
        sdpa_cus.proj_layer["value"]._weights.clone().detach().numpy(),
    )

    if add_bias:
        sdpa_pt.set_bias(
            sdpa_cus.proj_layer["query"]._bias.clone().detach().numpy(),
            sdpa_cus.proj_layer["key"]._bias.clone().detach().numpy(),
            sdpa_cus.proj_layer["value"]._bias.clone().detach().numpy(),
        )
        sdpa_tf.set_bias(
            sdpa_cus.proj_layer["query"]._bias.clone().detach().numpy(),
            sdpa_cus.proj_layer["key"]._bias.clone().detach().numpy(),
            sdpa_cus.proj_layer["value"]._bias.clone().detach().numpy(),
        )

    # Initialize loss functions and optimizers for each framework
    loss = RMSELoss()
    loss_torch = torch.nn.MSELoss()
    loss_tf = tf.keras.losses.MeanSquaredError()

    optimizer = get_optimizer("adam")(learning_rate=learning_rate)
    optimizer_tf = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    pt_training_params = [sdpa_pt.W_key, sdpa_pt.W_query, sdpa_pt.W_value]
    pt_training_params.extend(
        [sdpa_pt.b_query, sdpa_pt.b_key, sdpa_pt.b_value] if add_bias else []
    )
    optimizer_torch = torch.optim.Adam(
        params=pt_training_params,
        lr=learning_rate,
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
    optimizer.set_cur_epoch(1)
    sdpa_cus.backprop(dL, optimizer)

    cost_pt.backward()
    optimizer_torch.step()
    optimizer_torch.zero_grad()

    trainable_variables = [sdpa_tf.W_query, sdpa_tf.W_key, sdpa_tf.W_value]
    trainable_variables.extend(
        [sdpa_tf.b_query, sdpa_tf.b_key, sdpa_tf.b_value] if add_bias else []
    )
    grads = tape.gradient(cost_tf, trainable_variables)
    # if DEBUG:
    #     print("Grads (TF):", grads)
    optimizer_tf.apply_gradients(zip(grads, trainable_variables, strict=False))

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
    ), f"{get_output_template('tf')}"
    assert check_closeness(
        output_cus.detach().numpy(), output_pt.detach().numpy()
    ), f"{get_output_template('pt')}"
    if add_bias:
        for key, b_tf, b_pt in zip(
            ["query", "key", "value"],
            [sdpa_tf.b_query, sdpa_tf.b_key, sdpa_tf.b_value],
            [sdpa_pt.b_query, sdpa_pt.b_key, sdpa_pt.b_value],
            strict=False,
        ):
            assert check_closeness(
                sdpa_cus.proj_layer[key]._bias.detach().numpy(), b_tf.numpy()
            ), f"{get_bias_template('tf')}"
            assert check_closeness(
                sdpa_cus.proj_layer[key]._bias.detach().numpy(),
                b_pt.detach().numpy(),
            ), f"{get_bias_template('pt')}"

    # Check closeness of weights
    for key, W_tf, W_pt in zip(
        ["query", "key", "value"],
        [sdpa_tf.W_query, sdpa_tf.W_key, sdpa_tf.W_value],
        [sdpa_pt.W_query, sdpa_pt.W_key, sdpa_pt.W_value],
        strict=False,
    ):
        assert check_closeness(
            sdpa_cus.proj_layer[key]._weights.detach().numpy(), W_tf.numpy()
        ), f"{get_weight_template('tf')}"
        assert check_closeness(
            sdpa_cus.proj_layer[key]._weights.detach().numpy(),
            W_pt.detach().numpy(),
        ), f"{get_weight_template('pt')}"
