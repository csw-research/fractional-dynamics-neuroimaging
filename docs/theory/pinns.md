## Physics-Informed Neural Networks (PINNs) for Fractional PDEs

## Overview

Physics-Informed Neural Networks (PINNs) combine:
1. **Data-driven learning** (neural networks fit observations)
2. **Physics-based constraints** (PDEs encoded as soft constraints)

This hybrid approach enables:
- Learning from sparse, noisy data
- Enforcing known physical laws
- Solving inverse problems (parameter estimation)
- Uncertainty quantification

## Architecture

### Basic PINN Structure

**Neural Network**: u_θ(x, t) approximates solution u(x, t)

**Input**: Spatiotemporal coordinates (x, t)
**Output**: Field values u(x, t)
**Parameters**: θ = {W^(ℓ), b^(ℓ)} (weights and biases)

Typical architecture:
```
(x, t) → [Dense(64)] → [Tanh] → [Dense(64)] → [Tanh] → ... → [Dense(1)] → u(x, t)
```

**Activation functions**:
- Tanh: Smooth, bounded, works well for PDEs
- GELU: Modern alternative, smoother than ReLU
- Sin: For high-frequency solutions (periodic problems)

### Automatic Differentiation

**Key insight**: Compute PDE residuals via autograd

For u_θ(x, t), derivatives are obtained by backpropagation:
```python
u = model(x, t)
u_t = autograd.grad(u, t)  # ∂u/∂t
u_x = autograd.grad(u, x)  # ∂u/∂x
u_xx = autograd.grad(u_x, x)  # ∂²u/∂x²
```

**For fractional derivatives**: Use numerical approximations (Caputo L1/L2)
- Requires sorting data by time
- Compute fractional derivative on network outputs
- Integrate into computational graph

## Loss Function

PINN loss combines multiple objectives:

```
L_total = λ_data L_data + λ_PDE L_PDE + λ_BC L_BC + λ_IC L_IC
```

### 1. Data Loss

Match observations at measurement points:
```
L_data = 1/N_data Σᵢ |u_θ(xᵢ, tᵢ) - uᵢ^obs|²
```

### 2. PDE Loss (Physics)

Enforce fractional PDE at collocation points:
```
L_PDE = 1/N_PDE Σⱼ |D^α u_θ(xⱼ, tⱼ) - f(xⱼ, tⱼ, u_θ, ...)|²
```

**Example (fractional diffusion)**:
```
Residual = D^α_t u - D (-Δ)^{β/2} u
```

### 3. Boundary Conditions Loss

**Dirichlet**: u = g on ∂Ω
```
L_BC = 1/N_BC Σₖ |u_θ(x_BC,k) - g(x_BC,k)|²
```

**Neumann**: ∂u/∂n = h on ∂Ω
```
L_BC = 1/N_BC Σₖ |∂u_θ/∂n(x_BC,k) - h(x_BC,k)|²
```

### 4. Initial Conditions Loss

Enforce u(x, 0) = u₀(x):
```
L_IC = 1/N_IC Σₘ |u_θ(xₘ, 0) - u₀(xₘ)|²
```

## Training Strategy

### 1. Collocation Point Sampling

**Uniform**: Regular grid in (x, t) space
- Simple, systematic coverage
- May miss complex regions

**Random**: Latin hypercube sampling
- Better exploration
- Avoids structured artifacts

**Adaptive**: Residual-based refinement
- Add points where |PDE residual| is large
- Focuses compute on difficult regions

### 2. Loss Weighting

Challenge: Balance multiple loss components

**Options**:
1. **Manual tuning**: λ_data = 1, λ_PDE = 0.1, ...
2. **Adaptive weights**: Adjust based on gradient magnitudes
3. **Curriculum learning**: Gradually increase λ_PDE

**Gradient-based balancing**:
```python
# Compute gradient norms
∇_data = ||∂L_data/∂θ||
∇_PDE = ||∂L_PDE/∂θ||

# Set weights inversely proportional
λ_PDE ∝ 1/∇_PDE
```

Ensures all loss terms contribute equally to gradient updates.

### 3. Optimization

**Adam**: Default choice
- Learning rate: 1e-3 to 1e-4
- Fast convergence
- Works well for most problems

**L-BFGS**: Second-order method
- Use after Adam pre-training
- Better final accuracy
- More expensive per iteration

**Training procedure**:
```python
# Stage 1: Adam (10,000 epochs)
optimizer = Adam(lr=1e-3)
for epoch in range(10000):
    loss = compute_loss()
    loss.backward()
    optimizer.step()

# Stage 2: L-BFGS (1,000 iterations)
optimizer = LBFGS()
optimizer.step(closure)
```

## Fractional PINNs

### Temporal Fractional Derivatives

For PDE: D^α_t u = f(x, t, u, ...)

**Implementation**:
1. Sort collocation points by time: t₁ < t₂ < ... < t_N
2. Compute u_θ at all sorted points
3. Apply Caputo L1/L2 scheme to get D^α u_θ
4. Compute residual: r = D^α u_θ - f(...)
5. Backpropagate through entire pipeline

**Challenge**: Non-local nature requires evaluating at all previous times

**Solution**: Use autograd-compatible fractional derivative:
```python
def caputo_derivative_autograd(u, alpha, dt):
    # Detach, compute derivative, reattach to graph
    u_detached = u.detach()
    du_frac = caputo_L1_scheme(u_detached, alpha, dt)

    # Enable gradients
    du_frac.requires_grad = True
    return du_frac
```

### Spatial Fractional Derivatives

For fractional Laplacian: (-Δ)^{s} u

**FFT-based (for periodic BC)**:
```python
def fractional_laplacian_fft(u, alpha, dx):
    u_hat = fft(u)
    k = fftfreq(n, dx)
    multiplier = |k|^alpha
    result = ifft(multiplier * u_hat)
    return result.real
```

**Matrix-based (small problems)**:
- Construct L^α via eigendecomposition
- L^α = V Λ^α V^T where L = V Λ V^T

### Learning Fractional Order

**Parameterize α as trainable**:
```python
class FractionalPINN(nn.Module):
    def __init__(self):
        self.network = ...
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, t):
        u = self.network(x, t)
        # Use self.alpha in fractional derivative
        return u
```

**Constrain to (0, 1)**: α_actual = sigmoid(α_raw)

**Regularization**: Add prior if known range

**Result**: Joint optimization over u_θ and α

## Uncertainty Quantification

### 1. Bayesian PINNs

Place prior over network weights: p(θ)

**Posterior via variational inference**:
```
q(θ) ≈ p(θ | data, PDE)
```

**Loss function** (ELBO):
```
L = E_q[L_data + L_PDE] + KL(q || p)
```

**Prediction with uncertainty**:
```python
# Sample from posterior
for _ in range(N_samples):
    θ ~ q(θ)
    u_pred = model_θ(x, t)
    predictions.append(u_pred)

# Epistemic uncertainty
mean = mean(predictions)
std = std(predictions)
```

### 2. Monte Carlo Dropout

**Keep dropout active during inference**:
```python
model.train()  # Enable dropout
predictions = [model(x) for _ in range(100)]
mean = torch.mean(predictions)
std = torch.std(predictions)
```

Simpler than full Bayesian approach, often effective.

### 3. Ensemble Methods

Train multiple PINNs with different:
- Random initializations
- Collocation point samples
- Architectures

**Ensemble prediction**:
```python
predictions = [model_i(x) for model_i in ensemble]
mean = mean(predictions)
std = std(predictions)  # Model uncertainty
```

## Advanced Topics

### Multi-Fidelity Learning

Combine low-fidelity (cheap) and high-fidelity (expensive) data:

```
u_high(x) = u_low(x) + Δu(x)
```

**Training**:
1. Pre-train on abundant low-fidelity data
2. Fine-tune correction Δu on sparse high-fidelity data

**Application**: Transfer learning across subjects/resolutions

### Operator Learning (DeepONet)

Learn operator G: u₀ → u(t) mapping initial conditions to solutions

**Architecture**:
- **Branch network**: Encodes input function u₀(x)
- **Trunk network**: Encodes query points (x, t)
- **Output**: G(u₀)(x, t) = branch · trunk

**Advantage**: Generalize to new initial conditions without retraining

### Adaptive Refinement

**Residual-based sampling**:
```python
# Compute PDE residuals
residuals = compute_pde_residuals(collocation_points)

# Sample new points near high residuals
probs = |residuals| / sum(|residuals|)
new_points = sample(probs, N_new)

# Add to collocation set
collocation_points.extend(new_points)
```

**Iterative refinement**: Alternate training and point addition

## Applications to Neuroimaging

### 1. Fractional BOLD Modeling

**PDE**: D^α s(t) = ε u(t) - κ s(t) - γ(f(t) - 1)

**PINN advantages**:
- Learn α from data
- Incorporate sparse fMRI measurements
- Uncertainty in hemodynamic parameters

### 2. Anomalous Diffusion in dMRI

**PDE**: ∂C/∂t = D_α (-Δ)^{α/2} C

**Inverse problem**: Estimate D_α and α from diffusion-weighted images

**PINN workflow**:
1. Input: Multi-shell dMRI (different b-values)
2. Network: C_θ(x, t; b)
3. Loss: Match signal, satisfy fractional diffusion
4. Output: Spatially-varying D_α(x), α(x)

### 3. Parameter Estimation

**Forward problem**: Given PDE + parameters → solution
**Inverse problem**: Given solution (data) → parameters

**PINN for inverse**:
1. Parameterize unknown quantities (α, D, etc.)
2. Train network to fit data while satisfying PDE
3. Learned parameters are model outputs

**Example**: Estimate α(x) mapping brain regions to fractional orders

## Implementation Tips

### 1. Initialization

**Xavier/He initialization**: Standard for deep networks

**Pre-training**: First train on data only (L_data), then add PDE loss

### 2. Normalization

**Input normalization**: Scale (x, t) to [-1, 1]
```python
x_norm = (x - x_min) / (x_max - x_min) * 2 - 1
```

**Output normalization**: If u has known range, normalize

### 3. Convergence Monitoring

**Track all loss components separately**:
```python
losses = {
    'total': [],
    'data': [],
    'pde': [],
    'bc': [],
    'ic': []
}
```

**Check PDE residuals**: Should decrease over training

**Validation**: Hold out data points, check generalization

### 4. Common Issues

**Problem**: PDE loss dominates, ignores data
**Solution**: Increase λ_data or use adaptive weighting

**Problem**: Network overfits data, large PDE residuals
**Solution**: Increase λ_PDE, add more collocation points

**Problem**: Training stagnates
**Solution**: Switch to L-BFGS, increase learning rate, check gradients

## References

1. **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019)**. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

2. **Pang, G., Lu, L., & Karniadakis, G. E. (2019)**. fPINNs: Fractional physics-informed neural networks. *SIAM Journal on Scientific Computing*, 41(4), A2603-A2626.

3. **Wang, S., Teng, Y., & Perdikaris, P. (2021)**. Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing*, 43(5), A3055-A3081.

4. **Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021)**. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. *Nature Machine Intelligence*, 3(3), 218-229.

## Summary

**PINNs for fractional PDEs**:
- Encode fractional derivatives in loss function
- Use autograd + numerical schemes (Caputo L1/L2)
- Learn fractional orders as parameters
- Quantify uncertainty via Bayesian or ensemble methods

**Key advantages**:
- Work with sparse, noisy neuroimaging data
- Incorporate physical knowledge (conservation, causality)
- Solve inverse problems (parameter estimation)
- Flexible, end-to-end differentiable

**Best practices**:
- Normalize inputs/outputs
- Balance loss components (adaptive weights)
- Use two-stage optimization (Adam → L-BFGS)
- Monitor PDE residuals
- Validate on held-out data
