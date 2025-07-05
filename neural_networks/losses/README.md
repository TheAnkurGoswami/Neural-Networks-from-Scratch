# Loss Functions

This document provides an overview of the loss functions implemented in this module, including their forward and backward pass formulas.

## CrossEntropyLoss

Computes the cross-entropy loss between predicted probabilities `y_pred` and true labels `y_true`.

### Forward Pass
For a single sample `i`, the cross-entropy loss `L_i` is:
```latex
L_i = - \sum_{c=1}^{C} y_{true_{ic}} \log(y_{pred_{ic}})
```
where `C` is the number of classes.

The total loss `L_CE` is the average over all `N` samples in the batch:
```latex
L_{CE} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{true_{ic}} \log(y_{pred_{ic}})
```

### Backward Pass
The gradient of the loss `L` with respect to the predicted values `y_pred` (denoted \(\frac{\partial L}{\partial y_{pred}}\)) is computed as:
```latex
\frac{\partial L}{\partial y_{pred}} = - \frac{y_{true}}{y_{pred}} \cdot \frac{1}{N}
```
*(Note: The factor \(\frac{1}{N}\), where N is the batch size, is applied in the implementation, derived from the derivative of the averaged forward loss.)*

## MSELoss (Mean Squared Error)

### Forward Pass
The Mean Squared Error `L_MSE` is calculated as:
```latex
L_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_{pred_i} - y_{true_i})^2
```
where:
- `N` is the total number of elements in `y_true`.
- `y_pred_i` is the i-th predicted value.
- `y_true_i` is the i-th true value.

### Backward Pass
The gradient \(\frac{\partial L_{MSE}}{\partial y_{pred}}\) is computed as:
```latex
\frac{\partial L_{MSE}}{\partial y_{pred}} = \frac{2}{N} (y_{pred} - y_{true})
```
where `N` is the total number of elements in `y_true`.

## RMSELoss (Root Mean Squared Error)

RMSE is the square root of MSE.

### Forward Pass
The Root Mean Squared Error `L_RMSE` is calculated as the square root of the Mean Squared Error `L_MSE`:
```latex
L_{RMSE} = \sqrt{L_{MSE}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_{pred_i} - y_{true_i})^2}
```
where `N` is the total number of elements.

### Backward Pass
The gradient is derived using the chain rule:
\( \frac{\partial L_{RMSE}}{\partial y_{pred}} = \frac{\partial L_{RMSE}}{\partial L_{MSE}} \times \frac{\partial L_{MSE}}{\partial y_{pred}} \)

Since \( \frac{\partial L_{RMSE}}{\partial L_{MSE}} = \frac{1}{2 \sqrt{L_{MSE}}} = \frac{1}{2 L_{RMSE}} \),
the gradient becomes:
```latex
\frac{\partial L_{RMSE}}{\partial y_{pred}} = \frac{1}{2 L_{RMSE}} \frac{\partial L_{MSE}}{\partial y_{pred}}
```

## RMSELossV2 (Root Mean Squared Error - Direct Version)

This version calculates RMSE and its gradient directly.

### Forward Pass
The RMSE `L_RMSE` is calculated directly as:
```latex
L_{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_{pred_i} - y_{true_i})^2}
```
where `N` is the total number of elements in `y_true`.

### Backward Pass
The gradient \(\frac{\partial L_{RMSE}}{\partial y_{pred}}\) is:
```latex
\frac{\partial L_{RMSE}}{\partial y_{pred}} = \frac{1}{N \cdot L_{RMSE}} (y_{pred} - y_{true})
```
where:
- `N` is the total number of elements.
- `L_RMSE` is the computed RMSE loss from the forward pass.
