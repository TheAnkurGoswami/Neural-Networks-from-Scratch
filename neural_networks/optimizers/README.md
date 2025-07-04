# Optimizers

This document provides an overview of the optimization algorithms implemented in this module, detailing their update rules. The parameter \( \theta \) is updated as \( \theta_t = \theta_{t-1} - \text{update_value} \).

## Adam (Adaptive Moment Estimation)

Adam computes adaptive learning rates for each parameter.

Let \( g_t \) be the derivative (gradient) at timestep \( t \), \( \eta \) be the learning rate, \( \beta_1, \beta_2 \) be decay rates, \( m_t \) be the first moment vector (mean), and \( v_t \) be the second moment vector (uncentered variance). The timestep for bias correction is denoted by \( t \) (effectively `self._epoch` in the code for Adam).

The update rules are:
1. Update biased first moment estimate:
   ```latex
   m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
   ```
2. Update biased second raw moment estimate:
   ```latex
   v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
   ```
3. Compute bias-corrected first moment estimate:
   ```latex
   \hat{m}_t = \frac{m_t}{1 - \beta_1^t}
   ```
4. Compute bias-corrected second raw moment estimate:
   ```latex
   \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
   ```
5. Compute parameter update value:
   ```latex
   \text{update_value} = \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
   ```

## RMSProp

RMSProp divides the learning rate for a weight by a running average of the magnitudes of recent gradients for that weight.

Let \( g_t \) be the derivative (gradient) at timestep \( t \), \( \eta \) be the learning rate, \( \rho \) be the discounting factor, and \( E[g^2]_t \) be the moving average of squared gradients.

The update rule is:
1. Update moving average of squared gradients:
   ```latex
   E[g^2]_t = \rho E[g^2]_{t-1} + (1 - \rho) g_t^2
   ```
2. Compute parameter update value:
   ```latex
   \text{update_value} = \frac{\eta}{\sqrt{E[g^2]_t} + \epsilon} g_t
   ```

## SGD (Stochastic Gradient Descent)

Optionally includes momentum.

Let \( g_t \) be the derivative (gradient) at timestep \( t \), \( \eta \) be the learning rate, and \( \beta \) be the momentum factor.

If momentum (\( \beta > 0 \)):
1. Update velocity:
   ```latex
   v_t = \beta v_{t-1} + g_t
   ```
2. Compute parameter update value:
   ```latex
   \text{update_value} = \eta v_t
   ```

If no momentum (\( \beta = 0 \)):
1. Compute parameter update value:
   ```latex
   \text{update_value} = \eta g_t
   ```
