from typing import Dict

import numpy as np
import pytest
import tensorflow as tf
import torch

from neural_networks.attention.multihead_attention import MultiHeadAttention
from neural_networks.attention.projection import Projection
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
    # ScaledDotProductAttentionTensorflow,
    check_closeness,
)

DEBUG = True


# @pytest.mark.parametrize("d_model", [3, 5, 10])
# @pytest.mark.parametrize("batch_size", [1, 2, 64])
# @pytest.mark.parametrize("dim_kqv", [3, 7])
# @pytest.mark.parametrize("add_bias", [True, False])
# def test_scaled_dot_product_attention(add_bias: bool):
#     d_model = 3
#     seq_len = 5
#     batch_size = 1
#     learning_rate = 0.001

#     x = np.random.randint(
#         low=0, high=10, size=(batch_size, seq_len, d_model)
#     ).astype(np.float32)
#     y = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)

#     # Convert input to PyTorch tensor
#     x_torch = torch.tensor(x.astype(np.float32))
#     y_torch = torch.tensor(y.astype(np.float32))
#     # Convert input to TensorFlow tensor
#     x_tf = tf.constant(x.astype(np.float32))
#     y_tf = tf.constant(y.astype(np.float32))

#     proj_layer: Dict[str, Projection] = {}

#     for param in ["query", "key", "value"]:
#         proj_layer[param] = Projection(
#             in_features=d_model,
#             out_features=d_model,
#             add_bias=add_bias,
#         )

#     sdpa_cus = ScaledDotProductAttention()

#     sdpa_pt = ScaledDotProductAttentionPytorch(
#         d_model=d_model, dim_k=d_model, dim_v=d_model, add_bias=add_bias
#     )
#     sdpa_pt.set_weights(
#         proj_layer["query"]._weights.clone().detach().numpy(),
#         proj_layer["key"]._weights.clone().detach().numpy(),
#         proj_layer["value"]._weights.clone().detach().numpy(),
#     )

#     # sdpa_tf = ScaledDotProductAttentionTensorflow(
#     #     d_model=d_model, dim_k=d_model, dim_v=d_model, add_bias=add_bias
#     # )
#     # sdpa_tf.set_weights(
#     #     proj_layer["query"]._weights.clone().detach().numpy(),
#     #     proj_layer["key"]._weights.clone().detach().numpy(),
#     #     proj_layer["value"]._weights.clone().detach().numpy(),
#     # )

#     if add_bias:
#         sdpa_pt.set_bias(
#             proj_layer["query"]._bias.clone().detach().numpy(),
#             proj_layer["key"]._bias.clone().detach().numpy(),
#             proj_layer["value"]._bias.clone().detach().numpy(),
#         )
#         # sdpa_tf.set_bias(
#         #     proj_layer["query"]._bias.clone().detach().numpy(),
#         #     proj_layer["key"]._bias.clone().detach().numpy(),
#         #     proj_layer["value"]._bias.clone().detach().numpy(),
#         # )

#     # Initialize loss functions and optimizers for each framework
#     loss = RMSELoss()
#     loss_torch = torch.nn.MSELoss()
#     # loss_tf = tf.keras.losses.MeanSquaredError()

#     optimizer = get_optimizer("adam")(learning_rate=learning_rate)
#     # optimizer_tf = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#     pt_training_params = [sdpa_pt.W_key, sdpa_pt.W_query, sdpa_pt.W_value]
#     pt_training_params.extend(
#         [sdpa_pt.b_query, sdpa_pt.b_key, sdpa_pt.b_value] if add_bias else []
#     )
#     optimizer_torch = torch.optim.Adam(
#         params=pt_training_params,
#         lr=learning_rate,
#     )

#     projections = []
#     for param in ["query", "key", "value"]:
#         print(param)
#         dZ_cus = proj_layer[param].forward(x_torch.clone().detach())
#         projections.append(dZ_cus)
#     output_cus = sdpa_cus.forward(*projections)
#     cost_cus = loss.forward(output_cus, y_torch)

#     output_pt = sdpa_pt.forward(x_torch)
#     cost_pt = torch.sqrt(loss_torch(output_pt, y_torch))

#     # with tf.GradientTape() as tape:
#     #     output_tf = sdpa_tf.forward(x_tf)
#     #     cost_tf = tf.sqrt(loss_tf(output_tf, y_tf))

#     # Backward pass and optimization
#     dL = loss.backprop()
#     optimizer.set_cur_epoch(1)
#     dQ_cus, dK_cus, dV_cus = sdpa_cus.backprop(dL)

#     for param, dZ_cus in zip(
#         ["query", "key", "value"], [dQ_cus, dK_cus, dV_cus], strict=False
#     ):
#         proj_layer[param].backprop(dZ_cus, optimizer)

#     cost_pt.backward()
#     optimizer_torch.step()
#     # print("pt", sdpa_pt.scores.grad)
#     # if add_bias:
#     # for param, dZ_cus in zip(
#     #     ["query", "key", "value"],
#     # [sdpa_pt.b_query, sdpa_pt.b_key, sdpa_pt.b_value], strict=False
#     # ):
#     #     print("pt", param, dZ_cus.grad)
#     optimizer_torch.zero_grad()

#     # trainable_variables = [sdpa_tf.W_query, sdpa_tf.W_key, sdpa_tf.W_value]
#     # trainable_variables.extend(
#     #     [sdpa_tf.b_query, sdpa_tf.b_key, sdpa_tf.b_value,
#     # sdpa_tf.scores] if add_bias else []
#     # )
#     # grads = tape.gradient(cost_tf, trainable_variables)
#     # if DEBUG:
#     #     print("Grads (TF):", grads)
#     # optimizer_tf.apply_gradients(zip(grads, trainable_variables,
#     # strict=False))

#     # print("Loss (Custom):", cost_cus)
#     # print("Loss (PT):", cost_pt.item())
#     # print("Loss (TF):", cost_tf.numpy())
#     # assert check_closeness(
#     #     cost_cus.detach().numpy(), cost_tf
#     # ), f"{get_loss_template('tf')}"
#     assert check_closeness(
#         cost_cus.detach().numpy(), cost_pt.item()
#     ), f"{get_loss_template('pt')}"

#     # Check closeness of outputs
#     # assert check_closeness(
#     #     output_cus.detach().numpy(), output_tf.numpy()
#     # ), f"{get_output_template('tf')}"
#     assert check_closeness(
#         output_cus.detach().numpy(), output_pt.detach().numpy()
#     ), f"{get_output_template('pt')}"
#     if add_bias:
#         for key, b_pt in zip(
#             ["query", "key", "value"],
#             # [sdpa_tf.b_query, sdpa_tf.b_key, sdpa_tf.b_value],
#             [sdpa_pt.b_query, sdpa_pt.b_key, sdpa_pt.b_value],
#             strict=False,
#         ):
#             # print(key, '>', proj_layer[key]._bias.detach().numpy(),
#             # b_tf.numpy(), b_pt.detach().numpy())
#             # assert check_closeness(
#             #     proj_layer[key]._bias.detach().numpy(), b_tf.numpy()
#             # ), f"{get_bias_template('tf')}"
#             assert check_closeness(
#                 proj_layer[key]._bias.detach().numpy(),
#                 b_pt.detach().numpy(),
#             ), f"{get_bias_template('pt')}"

#     # Check closeness of weights
#     for key, W_pt in zip(
#         ["query", "key", "value"],
#         # [sdpa_tf.W_query, sdpa_tf.W_key, sdpa_tf.W_value],
#         [sdpa_pt.W_query, sdpa_pt.W_key, sdpa_pt.W_value],
#         strict=False,
#     ):
#         # print(key, proj_layer[key]._weights.detach().numpy(),
#         #  W_tf.numpy(), W_pt.detach().numpy())
#         # assert check_closeness(
#         #     proj_layer[key]._weights.detach().numpy(), W_tf.numpy()
#         # ), f"{get_weight_template('tf')}"
#         assert check_closeness(
#             proj_layer[key]._weights.detach().numpy(),
#             W_pt.detach().numpy(),
#         ), f"{get_weight_template('pt')}"
#     # assert False


# TODO: batch_size = 1
@pytest.mark.parametrize("batch_size", [2, 64])
def test_multi_head_attention(batch_size):
    d_model = 4
    seq_len = 5
    dim_kqv = d_model
    n_heads = 2
    learning_rate = 0.001

    x = np.random.randint(
        low=0, high=10, size=(batch_size, seq_len, d_model)
    ).astype(np.float32)
    y = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)

    # Convert input to PyTorch tensor
    x_torch = torch.tensor(x.astype(np.float32))
    y_torch = torch.tensor(y.astype(np.float32))

    mha_cus = MultiHeadAttention(
        d_model=d_model,
        num_heads=n_heads,
        dim_q=dim_kqv,
        dim_k=dim_kqv,
        dim_v=dim_kqv,
    )
    all_proj_wt = []
    all_proj_bias = []
    for projection in mha_cus.proj_layer.values():
        all_proj_wt.append(projection._weights.T)
        all_proj_bias.append(projection._bias)

    all_proj_wt = torch.cat(all_proj_wt, dim=0)
    all_proj_bias = torch.cat(all_proj_bias, dim=-1)

    in_proj_wt = torch.vstack(list(all_proj_wt))
    in_proj_bias = torch.hstack(list(all_proj_bias))

    mha_pt = torch.nn.MultiheadAttention(
        embed_dim=d_model, num_heads=n_heads, batch_first=True
    )

    mha_pt.in_proj_weight = torch.nn.Parameter(in_proj_wt.detach().clone())
    mha_pt.in_proj_bias = torch.nn.Parameter(in_proj_bias.detach().clone())
    mha_pt.out_proj.weight = torch.nn.Parameter(
        mha_cus.out_proj._weights.T.detach().clone()
    )
    mha_pt.out_proj.bias = torch.nn.Parameter(
        mha_cus.out_proj._bias.detach().clone()
    )
    output_cus = mha_cus.forward(x_torch)
    output_pt = mha_pt.forward(x_torch, x_torch, x_torch, need_weights=True)[0]

    # print("before", in_proj_wt, mha_pt.in_proj_weight)
    # Initialize loss functions and optimizers for each framework
    loss = RMSELoss()
    loss_torch = torch.nn.MSELoss()

    optimizer = get_optimizer("adam")(learning_rate=learning_rate)
    # pt_training_params = [sdpa_pt.W_key, sdpa_pt.W_query, sdpa_pt.W_value]

    optimizer_torch = torch.optim.Adam(
        params=mha_pt.parameters(),
        lr=learning_rate,
    )

    cost_cus = loss.forward(output_cus, y_torch)
    output_pt.retain_grad()
    cost_pt = torch.sqrt(loss_torch(output_pt, y_torch))

    # print("cost", cost_cus, cost_pt)

    # Backward pass and optimization
    dL = loss.backprop()
    # print("dL", dL)
    optimizer.set_cur_epoch(1)
    mha_cus.backprop(dL, optimizer)

    cost_pt.backward()
    # # print("DL", output_pt.grad)
    # print("mha_pt.out", mha_pt.out_proj.weight.grad)
    # print("mha_pt.out", mha_pt.out_proj.bias.grad)
    print("mha_pt.in_proj_weight", mha_pt.in_proj_weight.grad)
    optimizer_torch.step()
    optimizer_torch.zero_grad()

    # Check closeness of outputs
    assert check_closeness(
        output_cus.detach().numpy(), output_pt.detach().numpy()
    ), f"{get_output_template('pt')}"

    assert check_closeness(
        cost_cus.detach().numpy(), cost_pt.item()
    ), f"{get_loss_template('pt')}"

    # print("after", in_proj_wt, mha_pt.in_proj_weight)
    # Check closeness of weights

    all_proj_wt = []
    all_proj_bias = []
    for projection in mha_cus.proj_layer.values():
        all_proj_wt.append(projection._weights.T)
        all_proj_bias.append(projection._bias)

    all_proj_wt = torch.cat(all_proj_wt, dim=0)
    all_proj_bias = torch.cat(all_proj_bias, dim=-1)

    in_proj_wt = torch.vstack(list(all_proj_wt))
    in_proj_bias = torch.hstack(list(all_proj_bias))

    assert check_closeness(
        in_proj_wt.detach().numpy(),
        mha_pt.in_proj_weight.detach().numpy(),
    ), f"{get_weight_template('pt')}"

    assert check_closeness(
        in_proj_bias.detach().numpy(),
        mha_pt.in_proj_bias.detach().numpy(),
    ), f"{get_bias_template('pt')}"

    assert check_closeness(
        mha_cus.out_proj._weights.T.detach().clone(),
        mha_pt.out_proj.weight.detach().numpy(),
    ), f"{get_weight_template('pt')}"

    assert check_closeness(
        mha_cus.out_proj._bias.detach().clone(),
        mha_pt.out_proj.bias.detach().numpy(),
    ), f"{get_bias_template('pt')}"
