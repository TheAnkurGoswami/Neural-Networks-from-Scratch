# Activation Functions

This document provides an overview of the activation functions implemented in this module, including their forward and backward pass formulas.

## Identity

### Forward Pass
The output is the same as the input.
```latex
\text{Identity}(x) = x
```

### Backward Pass
The derivative of the Identity function is always 1.
```latex
\frac{d}{dx} \text{Identity}(x) = 1
```

## ReLU

### Forward Pass
The ReLU function is defined as:
```latex
\text{ReLU}(x) = \max(0, x)
```

### Backward Pass
The derivative of ReLU is:
```latex
\text{ReLU}'(x) = \begin{cases}
    1 & \text{if } x > 0 \\
    0 & \text{if } x \leq 0
\end{cases}
```

## Sigmoid

### Forward Pass
The sigmoid function is defined as:
```latex
\sigma(x) = \frac{1}{1 + e^{-x}}
```

### Backward Pass
The derivative is calculated using the formula:
```latex
\sigma'(x) = \sigma(x) (1 - \sigma(x))
```
where \(\sigma(x)\) is the activation value from the forward pass.

## Softmax

### Forward Pass
The Softmax function is defined as:
```latex
\text{Softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j} e^{x_j - \max(x)}}
```
Subtracting \(\max(x)\) from \(x\) improves numerical stability without changing the output.

### Backward Pass
Given the gradient of the loss w.r.t the Softmax output (\( \frac{\partial L}{\partial a} = dA \)), this computes the gradient of the loss w.r.t the Softmax input (\( \frac{\partial L}{\partial z} \)).
The calculation can be simplified to:
```latex
dZ_k = a_k \left( dA_k - \sum_j dA_j \cdot a_j \right)
```
So, \( dZ = \text{act} \odot (dA - \text{sum}(dA \odot \text{act}, \text{dim}, \text{keepdim=True})) \)
where \( \odot \) is element-wise multiplication.

*Note: Softmax also has a `derivative` method that computes the Jacobian matrix:*
```latex
J_{ij} = \begin{cases}
            a_i (1 - a_i) & \text{if } i = j \\
            -a_i a_j & \text{if } i \neq j
        \end{cases}
```

## Tanh

### Forward Pass
The tanh activation function is defined as:
```latex
\text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
```

### Backward Pass
The derivative is calculated using the formula:
```latex
\text{tanh}'(x) = 1 - \text{tanh}^2(x)
```
where \(\text{tanh}(x)\) is the activation value from the forward pass.
