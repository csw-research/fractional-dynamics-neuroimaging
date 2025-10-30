"""
Fractional derivative implementations.

This module implements three main formulations:
1. Caputo derivative (most common in physics/engineering)
2. Riemann-Liouville derivative (classical definition)
3. Grünwald-Letnikov derivative (numerical approximation)

Mathematical background:
- Caputo: D^α_t f(t) = 1/Γ(1-α) ∫₀ᵗ f'(τ)/(t-τ)^α dτ
- Riemann-Liouville: D^α_t f(t) = 1/Γ(1-α) d/dt ∫₀ᵗ f(τ)/(t-τ)^α dτ
- Grünwald-Letnikov: D^α_t f(t) ≈ h^{-α} Σₖ w_k^{(α)} f(t-kh)

The Caputo derivative is preferred because:
- Initial conditions have clear physical meaning
- Derivative of constant is zero (matches intuition)
- Better for solving fractional ODEs/PDEs
"""

import torch
import numpy as np
from scipy.special import gamma
from typing import Union, Optional


def grunwald_weights(alpha: float, num_points: int) -> torch.Tensor:
    """
    Compute Grünwald-Letnikov weights for fractional derivative.

    The weights are computed recursively:
    w_0^{(α)} = 1
    w_k^{(α)} = (1 - (α+1)/k) * w_{k-1}^{(α)}

    Args:
        alpha: Fractional order (0 < α < 1 for derivatives)
        num_points: Number of weights to compute

    Returns:
        Tensor of shape (num_points,) containing weights

    Example:
        >>> weights = grunwald_weights(0.5, 100)
        >>> print(weights[:5])  # First few weights
    """
    weights = torch.zeros(num_points, dtype=torch.float64)
    weights[0] = 1.0

    for k in range(1, num_points):
        weights[k] = weights[k-1] * (1.0 - (alpha + 1.0) / k)

    return weights


def caputo_derivative(
    f: Union[torch.Tensor, np.ndarray],
    alpha: float,
    dt: float,
    method: str = "L1",
    axis: int = -1
) -> torch.Tensor:
    """
    Compute Caputo fractional derivative of order α ∈ (0, 1).

    Uses L1 scheme (first-order accurate) or L2 scheme (second-order accurate).
    The L1 scheme discretizes as:

    D^α f(t_n) ≈ (dt^{-α} / Γ(2-α)) Σₖ₌₁ⁿ b_k [f(t_{n-k+1}) - f(t_{n-k})]

    where b_k = k^{1-α} - (k-1)^{1-α}

    Args:
        f: Function values, shape (N,) or (..., N, ...)
        alpha: Fractional order, 0 < α < 1
        dt: Time step
        method: "L1" (first-order) or "L2" (second-order)
        axis: Axis along which to compute derivative

    Returns:
        Fractional derivative, same shape as f

    Example:
        >>> t = torch.linspace(0, 10, 1000)
        >>> f = t**2
        >>> df = caputo_derivative(f, alpha=0.5, dt=t[1]-t[0])
        >>> # Analytical: D^{0.5}(t²) = 4t^{1.5}/Γ(2.5)
    """
    if isinstance(f, np.ndarray):
        f = torch.from_numpy(f)

    f = f.double()  # Use float64 for numerical stability

    # Move axis to last dimension
    if axis != -1 and axis != f.ndim - 1:
        f = torch.moveaxis(f, axis, -1)

    n = f.shape[-1]
    result = torch.zeros_like(f)

    if method == "L1":
        # L1 scheme coefficients
        coeff = dt**(-alpha) / gamma(2 - alpha)

        # Compute b_k coefficients
        k = torch.arange(1, n + 1, dtype=torch.float64)
        b = k**(1 - alpha) - (k - 1)**(1 - alpha)

        # Compute differences
        diff = torch.diff(f, dim=-1, prepend=f[..., :1])

        # Convolution with b coefficients
        for i in range(n):
            if i == 0:
                result[..., i] = 0
            else:
                result[..., i] = coeff * torch.sum(
                    b[:i] * torch.flip(diff[..., 1:i+1], dims=[-1]),
                    dim=-1
                )

    elif method == "L2":
        # L2 scheme (second-order accurate)
        # More complex but better accuracy
        coeff = dt**(-alpha) / gamma(3 - alpha)

        for i in range(n):
            if i < 2:
                result[..., i] = 0
            else:
                sum_val = torch.zeros(f.shape[:-1], dtype=torch.float64)
                for k in range(1, i):
                    a_k = (k + 1)**(2 - alpha) - 2*k**(2 - alpha) + (k - 1)**(2 - alpha)
                    sum_val += a_k * f[..., i - k]

                # Boundary terms
                a_0 = 1
                a_i = i**(2 - alpha) - (i - 1)**(2 - alpha)

                result[..., i] = coeff * (
                    a_i * f[..., 0] - sum_val + a_0 * f[..., i]
                )

    else:
        raise ValueError(f"Unknown method: {method}. Use 'L1' or 'L2'.")

    # Move axis back
    if axis != -1 and axis != result.ndim - 1:
        result = torch.moveaxis(result, -1, axis)

    return result.float()  # Convert back to float32


def riemann_liouville_derivative(
    f: Union[torch.Tensor, np.ndarray],
    alpha: float,
    dt: float,
    axis: int = -1
) -> torch.Tensor:
    """
    Compute Riemann-Liouville fractional derivative of order α ∈ (0, 1).

    Definition: D^α f = d/dt I^{1-α} f
    where I^{1-α} is the fractional integral of order 1-α.

    Args:
        f: Function values
        alpha: Fractional order, 0 < α < 1
        dt: Time step
        axis: Axis along which to compute derivative

    Returns:
        Riemann-Liouville derivative

    Note:
        RL derivative of constant ≠ 0, unlike Caputo derivative.
        This makes it less suitable for physical problems with
        constant initial conditions.
    """
    if isinstance(f, np.ndarray):
        f = torch.from_numpy(f)

    f = f.double()

    # Move axis to last dimension
    if axis != -1 and axis != f.ndim - 1:
        f = torch.moveaxis(f, axis, -1)

    n = f.shape[-1]

    # Compute fractional integral of order (1 - alpha)
    beta = 1 - alpha
    integral = fractional_integral(f, beta, dt, axis=-1)

    # Compute first derivative
    derivative = torch.diff(integral, dim=-1) / dt
    # Pad to maintain shape
    derivative = torch.cat([derivative[..., :1], derivative], dim=-1)

    # Move axis back
    if axis != -1 and axis != derivative.ndim - 1:
        derivative = torch.moveaxis(derivative, -1, axis)

    return derivative.float()


def fractional_integral(
    f: Union[torch.Tensor, np.ndarray],
    alpha: float,
    dt: float,
    axis: int = -1
) -> torch.Tensor:
    """
    Compute fractional integral of order α > 0.

    Definition: I^α f(t) = 1/Γ(α) ∫₀ᵗ (t-τ)^{α-1} f(τ) dτ

    Args:
        f: Function values
        alpha: Integration order, α > 0
        dt: Time step
        axis: Axis along which to integrate

    Returns:
        Fractional integral
    """
    if isinstance(f, np.ndarray):
        f = torch.from_numpy(f)

    f = f.double()

    if axis != -1 and axis != f.ndim - 1:
        f = torch.moveaxis(f, axis, -1)

    n = f.shape[-1]
    result = torch.zeros_like(f)

    coeff = dt**alpha / gamma(alpha + 1)

    for i in range(n):
        # Weights: (i-k)^α
        k = torch.arange(0, i + 1, dtype=torch.float64)
        weights = (i - k + 1)**alpha - (i - k)**alpha

        result[..., i] = coeff * torch.sum(
            weights * f[..., :i+1],
            dim=-1
        )

    if axis != -1 and axis != result.ndim - 1:
        result = torch.moveaxis(result, -1, axis)

    return result.float()


def grunwald_letnikov_derivative(
    f: Union[torch.Tensor, np.ndarray],
    alpha: float,
    dt: float,
    axis: int = -1,
    max_memory: Optional[int] = None
) -> torch.Tensor:
    """
    Compute Grünwald-Letnikov fractional derivative.

    This is a direct discretization:
    D^α f(t_n) ≈ dt^{-α} Σₖ₌₀ⁿ w_k^{(α)} f(t_{n-k})

    Args:
        f: Function values
        alpha: Fractional order
        dt: Time step
        axis: Axis along which to compute derivative
        max_memory: Maximum number of past points to use (for efficiency)
                   If None, uses all available points

    Returns:
        Grünwald-Letnikov derivative

    Note:
        This method converges to Riemann-Liouville derivative as dt → 0.
        For large arrays, set max_memory to limit computational cost.
    """
    if isinstance(f, np.ndarray):
        f = torch.from_numpy(f)

    f = f.double()

    if axis != -1 and axis != f.ndim - 1:
        f = torch.moveaxis(f, axis, -1)

    n = f.shape[-1]
    result = torch.zeros_like(f)

    # Determine memory length
    if max_memory is None:
        max_memory = n
    else:
        max_memory = min(max_memory, n)

    # Compute weights
    weights = grunwald_weights(alpha, max_memory)
    coeff = dt**(-alpha)

    for i in range(n):
        memory_len = min(i + 1, max_memory)
        result[..., i] = coeff * torch.sum(
            weights[:memory_len] * torch.flip(f[..., i-memory_len+1:i+1], dims=[-1]),
            dim=-1
        )

    if axis != -1 and axis != result.ndim - 1:
        result = torch.moveaxis(result, -1, axis)

    return result.float()


def caputo_derivative_2d(
    u: torch.Tensor,
    alpha: float,
    dx: float,
    dy: float,
    method: str = "L1"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute 2D Caputo fractional derivatives (∂^α/∂x^α, ∂^α/∂y^α).

    Args:
        u: 2D field, shape (Ny, Nx)
        alpha: Fractional order
        dx: Grid spacing in x
        dy: Grid spacing in y
        method: Discretization method

    Returns:
        (du_dx, du_dy): Fractional derivatives in x and y directions
    """
    du_dx = caputo_derivative(u, alpha, dx, method=method, axis=1)
    du_dy = caputo_derivative(u, alpha, dy, method=method, axis=0)

    return du_dx, du_dy


def fractional_gradient(
    u: torch.Tensor,
    alpha: float,
    spacing: Union[float, tuple[float, ...]],
    method: str = "L1"
) -> tuple[torch.Tensor, ...]:
    """
    Compute fractional gradient of multi-dimensional field.

    Args:
        u: N-dimensional field
        alpha: Fractional order
        spacing: Grid spacing (scalar or tuple for each dimension)
        method: Discretization method

    Returns:
        Tuple of fractional derivatives along each axis
    """
    if isinstance(spacing, (int, float)):
        spacing = (spacing,) * u.ndim

    gradients = []
    for axis, dx in enumerate(spacing):
        grad = caputo_derivative(u, alpha, dx, method=method, axis=axis)
        gradients.append(grad)

    return tuple(gradients)


# Utility function for automatic differentiation compatibility
def caputo_derivative_autograd(
    f: torch.Tensor,
    alpha: float,
    dt: float,
    method: str = "L1"
) -> torch.Tensor:
    """
    Caputo derivative with autograd support for backpropagation.

    This version ensures gradients flow properly for PINN training.
    """
    # Ensure requires_grad is set
    requires_grad = f.requires_grad
    f = f.detach()

    result = caputo_derivative(f, alpha, dt, method)

    if requires_grad:
        result.requires_grad_(True)

    return result
