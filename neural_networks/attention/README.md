# Attention Mechanisms

This document provides an overview of the attention mechanisms implemented in this module, including their core formulas and operational principles.

## Scaled Dot-Product Attention

This is a core component allowing the model to weigh the importance of different parts of the input sequence.

### Forward Pass
The attention is calculated as:
```latex
\text{scores} = Q K^T
```
```latex
\text{scaled\_scores} = \frac{\text{scores}}{\sqrt{d_k}}
```
```latex
\text{attention\_weights} = \text{softmax}(\text{scaled\_scores})
```
```latex
\text{output} = \text{attention\_weights} V
```
Where:
- `Q`, `K`, `V` are Query, Key, and Value matrices, respectively.
- `d_k` is the dimension of the Key vectors.
- `softmax` is the softmax function applied to the scaled scores.

### Backward Pass
Given `dA` (or \(\frac{\partial L}{\partial A}\)), the gradient of the loss `L` with respect to the attention output `A`, the gradients with respect to `Q`, `K`, and `V` (denoted `dQ_proj`, `dK_proj`, `dV_proj`) are computed. Let `S` represent `scaled_scores`.

1.  Gradient w.r.t. `V` (Value):
    ```latex
    dV_{\text{proj}} = \text{softmax}(S)^T dA
    ```

2.  Gradient w.r.t. softmax output (`softmax(S)`):
    ```latex
    d\text{softmax}(S) = dA V^T
    ```

3.  Gradient w.r.t. scaled scores (`S`), passing through `softmax.backprop`:
    Let \(dS' = d\text{softmax}(S)\).
    ```latex
    dS = \text{softmax.backprop}(dS')
    ```
    (This typically involves \( \text{softmax}(S) \odot (dS' - \text{sum}(dS' \odot \text{softmax}(S))) \), where \(\odot\) is element-wise multiplication).

4.  Gradient w.r.t. unscaled scores (undo scaling):
    ```latex
    d(QK^T) = \frac{dS}{\sqrt{d_k}}
    ```

5.  Gradient w.r.t. `Q_proj` (Query):
    ```latex
    dQ_{\text{proj}} = d(QK^T) K
    ```

6.  Gradient w.r.t. `K_proj` (Key):
    ```latex
    dK_{\text{proj}} = d(QK^T)^T Q
    ```

## Multi-Head Attention

Multi-Head Attention allows the model to jointly attend to information from different representation subspaces.

### Forward Pass
1. Input `X` is linearly projected to create Query (`Q`), Key (`K`), and Value (`V`) matrices for each head `i`: `Q_i`, `K_i`, `V_i`.
2. Each head computes attention using Scaled Dot-Product Attention:
   ```latex
   \text{head}_i = \text{Attention}(Q_i, K_i, V_i)
   ```
3. The outputs of all heads are concatenated: `Concat(head_1, ..., head_h)`.
4. The concatenated output is linearly projected by an output weight matrix `W^O`:
   ```latex
   \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
   ```

### Backward Pass
The backward pass involves:
1. Propagating the incoming gradient `dA` (gradient of loss w.r.t. MultiHead output) through the final output projection `W^O` to get the gradient w.r.t. the concatenated head outputs.
2. Splitting this gradient and propagating it through each attention head (`head_i`) to get gradients for `Q_i`, `K_i`, `V_i`.
3. Concatenating these head-specific gradients to get gradients for the full (pre-split) `Q`, `K`, `V` projections.
4. Propagating these gradients through the initial `Q`, `K`, `V` linear projection layers (which are instances of the `Projection` class described below). This updates the weights of these projection layers and computes the gradient w.r.t. the original input `X`. If `Q, K, V` derive from the same input `X`, their gradients w.r.t. `X` are summed.


## Projection Layer

A specialized `Dense` layer used for creating Query, Key, Value, and output projections in attention mechanisms. It primarily performs a linear transformation.

### Forward Pass
```latex
Z = XW + b
```
Where:
- `X` is the input.
- `W` is the weights matrix.
- `b` is the bias vector.
- `Z` is the linear output.
Typically, no activation function is applied directly within these projection layers.

### Backward Pass
Given `dZ` (or \(\frac{\partial L}{\partial Z}\)), the gradient of the loss `L` w.r.t the layer's linear output `Z`:

1. Gradient for weights `W` (denoted `dW`):
   Calculated from `X` and `dZ`. For inputs `X` of shape (batch, seq_len, in_feat) and `dZ` of shape (batch, seq_len, out_feat), `dW` is typically computed by summing `X^T dZ` contributions across the batch and sequence dimensions appropriately. The implementation details show summation over the batch dimension after an initial batched matrix multiply: `dW_batched = X^T dZ`, then `dW = sum(dW_batched, axis=0)`.
   ```latex
   dW = \sum_{\text{batch}} (X^T dZ) \quad (\text{conceptual, actual sum might involve sequence too})
   ```

2. Gradient for bias `b` (denoted `dB`):
   Summed over batch and sequence dimensions of `dZ`.
   ```latex
   dB = \sum_{\text{batch}} \sum_{\text{sequence}} dZ
   ```

3. Gradient for input `X` (denoted `dX`):
   ```latex
   dX = dZ W^T
   ```
The weights `W` and bias `b` are then updated using these gradients and an optimizer.
