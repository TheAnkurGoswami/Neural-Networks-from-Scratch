from typing import Union

import numpy as np

import tensorflow as tf
import torch as pt

# Define a type alias for various number types
NUMBER_TYPE = Union[np.ndarray, float, int, pt.Tensor, tf.Tensor]


def check_closeness(
    a: np.ndarray,
    b: np.ndarray,
    additional_checks: bool = True,
    tolerance: float = 1e-06,
) -> bool:
    """
    Check if two numpy arrays are close to each other within a certain
    tolerance.

    Parameters:
    a (np.ndarray): First array to compare.
    b (np.ndarray): Second array to compare.
    additional_checks (bool): If True, perform additional checks for closeness.
        Default is True.
    tolerance (float): Tolerance value for element-wise comparison.
        Default is 1e-06.

    Returns:
    bool: True if arrays are close to each other, False otherwise.
    """
    # Check if arrays are element-wise equal within a tolerance
    main_check = np.allclose(a, b)
    clip_const = 1e-06
    # print(a, b)
    # # Reverse clip values of 'a' if they are of the order of e-07 or lower
    a = np.where(np.abs(a) > clip_const, a, clip_const)
    b = np.where(np.abs(b) > clip_const, b, clip_const)
    # print(a, b)
    # Check if the absolute difference between arrays is within the tolerance
    other_check = np.abs(a - b) <= tolerance
    with np.errstate(divide="ignore", invalid="ignore"):
        # Calculate the minimum of the two arrays element-wise
        # min_arr = np.minimum(a, b)
        max_arr = np.maximum(a, b)
        clipped_diff = np.where(np.abs(a - b) > clip_const, np.abs(a - b), 0)
        # Calculate the percentage difference where max_arr is not zero
        percent_diff = np.average(
            np.where(max_arr != 0, clipped_diff / max_arr * 100, 0)
        )
        # Check if the average percentage difference is within 0.001%
        precent_check = percent_diff <= 0.001

    if additional_checks:
        # Return True if any of the checks pass
        return bool(main_check or np.all(other_check) or precent_check)

    # Return the result of the main check
    return main_check



class ScaledDotProductAttentionPytorch(pt.nn.Module):
    """
    PyTorch implementation of the Scaled Dot-Product Attention mechanism.
    """

    def __init__(self, d_model: int, dim_k: int, dim_v: int) -> None:
        """
        Initialize the ScaledDotProductAttention class.

        Parameters:
        d_model (int): The dimension of the model.
        dim_k (int): The dimension of the key.
        dim_v (int): The dimension of the value.
        """
        self.d_model = d_model
        self.dim_k = dim_k
        self.dim_v = dim_v


    def set_weights(self, W_query: pt.Tensor, W_key: pt.Tensor, W_value: pt.Tensor):
        """
        Set the weights for the attention mechanism.

        Parameters:
        W_query (torch.Tensor): The query weight matrix.
        W_key (torch.Tensor): The key weight matrix.
        W_value (torch.Tensor): The value weight matrix.
        """
        self.W_query = pt.tensor(W_query, requires_grad=True)
        self.W_key = pt.tensor(W_key, requires_grad=True)
        self.W_value = pt.tensor(W_value, requires_grad=True)

    def forward(self, inputs):
        """
        Forward pass of the attention mechanism.

        Parameters:
        inputs (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The output of the attention mechanism.
        """
        # Compute queries, keys, and values
        queries = pt.matmul(inputs, self.W_query)
        keys = pt.matmul(inputs, self.W_key)
        values = pt.matmul(inputs, self.W_value)

        # Compute attention scores
        self.scores = pt.matmul(
            queries, keys.transpose(-2, -1)) / (self.dim_k ** 0.5)
        self.scores.retain_grad()
        # print("PT Attention Scores", scores.shape, scores)
        # Apply softmax to get attention weights
        self.attention_weights = pt.softmax(self.scores, dim=-1)
        self.attention_weights.retain_grad()
        # print("PT Attention Weights", attention_weights.shape, attention_weights)
        # Compute the output
        output = pt.matmul(self.attention_weights, values)

        return output


class ScaledDotProductAttentionTensorflow(tf.Module):
    """
    TensorFlow implementation of the Scaled Dot-Product Attention mechanism.
    """

    def __init__(self, d_model: int, dim_k: int, dim_v: int) -> None:
        """
        Initialize the ScaledDotProductAttention class.

        Parameters:
        d_model (int): The dimension of the model.
        dim_k (int): The dimension of the key.
        dim_v (int): The dimension of the value.
        """
        self.d_model = d_model
        self.dim_k = dim_k
        self.dim_v = dim_v

    def set_weights(self, W_query, W_key, W_value):
        """
        Set the weights for the attention mechanism.

        Parameters:
        W_query (tf.Tensor): The query weight matrix.
        W_key (tf.Tensor): The key weight matrix.
        W_value (tf.Tensor): The value weight matrix.
        """
        self.W_query = tf.Variable(W_query)
        self.W_key = tf.Variable(W_key)
        self.W_value = tf.Variable(W_value)

    def forward(self, inputs):
        """
        Forward pass of the attention mechanism.

        Parameters:
        inputs (tf.Tensor): The input tensor.

        Returns:
        tf.Tensor: The output of the attention mechanism.
        """
        # Compute queries, keys, and values
        queries = tf.matmul(inputs, self.W_query)
        keys = tf.matmul(inputs, self.W_key)
        values = tf.matmul(inputs, self.W_value)

        # Compute attention scores
        scores = tf.matmul(
            queries, keys, transpose_b=True) / (self.dim_k ** 0.5)

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scores, axis=-1)

        # Compute the output
        output = tf.matmul(attention_weights, values)

        return output
