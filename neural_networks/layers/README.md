# Layers

This document provides an overview of the neural network layers implemented in this module, including their forward and backward pass formulas.

## Dense (Fully Connected Layer)

A fully connected neural network layer (Linear layer).

### Forward Pass
The layer performs a linear transformation of the input data \(X\) using a weights matrix \(W\) and an optional bias vector \(b\), followed by an activation function \(f\).

The operations are:
1. Linear transformation:
   ```latex
   Z = XW + b
   ```
2. Activation:
   ```latex
   A = f(Z)
   ```
Where:
- \( X \) is the input data.
- \( W \) is the weights matrix.
- \( b \) is the bias vector (if `add_bias` is True).
- \( Z \) is the linear output.
- \( A \) is the activation output.
- \( f \) is the activation function.

### Backward Pass
Calculates gradients with respect to inputs (\( dX \)), weights (\( dW \)), and bias (\( dB \)), and updates weights and bias.

The steps are:
1. Compute gradient with respect to the output of the linear part (before activation):
   ```latex
   dZ = dA \odot f'(Z)
   ```
   Where \( dA = \frac{\partial L}{\partial A} \) is the gradient of the loss with respect to the layer's activation output, and \( f'(Z) = \frac{\partial A}{\partial Z} \) is the derivative of the activation function. Thus, \( dZ = \frac{\partial L}{\partial Z} \).

2. Compute gradient for weights:
   ```latex
   dW = X^T dZ
   ```

3. Compute gradient for bias (if bias is used):
   ```latex
   dB = \sum_{\text{batch}} dZ
   ```
   (sum over the batch dimension).

4. Compute gradient with respect to the input of this layer:
   ```latex
   dX = dZ W^T
   ```

The weights \(W\) and bias \(b\) are then updated using these gradients and an optimizer.
