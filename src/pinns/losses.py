"""
Loss functions for physics-informed neural networks.

Combines:
1. Data loss: Match observations
2. PDE loss: Satisfy fractional differential equations
3. Boundary loss: Satisfy boundary conditions
4. Initial condition loss: Match initial state
"""

import torch
import torch.nn as nn
from typing import Callable, Optional, Dict
import sys
sys.path.append('..')

from src.fractional.derivatives import caputo_derivative, caputo_derivative_autograd
from src.fractional.laplacian import fractional_laplacian_fft


def data_loss(
    u_pred: torch.Tensor,
    u_observed: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Mean squared error between predictions and observations.

    Args:
        u_pred: Predicted values, shape (N,)
        u_observed: Observed values, shape (N,)
        weights: Optional per-sample weights, shape (N,)

    Returns:
        Weighted MSE loss
    """
    squared_diff = (u_pred - u_observed) ** 2

    if weights is not None:
        squared_diff = squared_diff * weights

    return torch.mean(squared_diff)


def fractional_pde_loss(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    alpha: float,
    pde_func: Callable,
    dt: Optional[float] = None,
    dx: Optional[float] = None,
) -> torch.Tensor:
    """
    Loss for fractional PDE residual: D^α u - f(x, t, u) = 0

    Args:
        model: Neural network u(x, t)
        x: Spatial coordinates, shape (N, d)
        t: Time coordinates, shape (N, 1)
        alpha: Fractional order
        pde_func: Function f(x, t, u, grad_u, ...) defining PDE
        dt: Time step (if None, computed from t)
        dx: Spatial step (if None, computed from x)

    Returns:
        Mean squared PDE residual

    Example:
        >>> # Fractional diffusion: D^α_t u = D ∇²u
        >>> def pde(x, t, u, u_t_frac, u_xx):
        >>>     return u_t_frac - D * u_xx
        >>> loss = fractional_pde_loss(model, x, t, alpha=0.5, pde_func=pde)
    """
    # Combine inputs
    xt = torch.cat([x, t], dim=1)
    xt.requires_grad_(True)

    # Forward pass
    u = model(xt)

    # Compute derivatives via autograd
    # Spatial derivatives (classical)
    grad_u = torch.autograd.grad(
        u, xt,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

    u_x = grad_u[:, :-1]  # Spatial components
    u_t = grad_u[:, -1:]  # Time component

    # Second spatial derivatives
    u_xx = []
    for i in range(x.shape[1]):
        u_xx_i = torch.autograd.grad(
            u_x[:, i:i+1], xt,
            grad_outputs=torch.ones_like(u_x[:, i:i+1]),
            create_graph=True,
            retain_graph=True
        )[0][:, i:i+1]
        u_xx.append(u_xx_i)

    u_xx = torch.cat(u_xx, dim=1) if len(u_xx) > 1 else u_xx[0]

    # Fractional time derivative (requires special handling)
    # For now, approximate using Caputo on the neural network output
    # In practice, you'd sort by time and apply fractional derivative

    # Sort by time
    t_sorted, sort_indices = torch.sort(t.squeeze(), dim=0)
    u_sorted = u[sort_indices]

    if dt is None:
        dt = torch.mean(t_sorted[1:] - t_sorted[:-1]).item()

    # Compute fractional derivative
    # Note: This is approximate and works best for regularly-spaced time points
    u_t_frac_sorted = caputo_derivative_autograd(
        u_sorted.squeeze(),
        alpha,
        dt,
        method="L1"
    )

    # Unsort
    inverse_indices = torch.argsort(sort_indices)
    u_t_frac = u_t_frac_sorted[inverse_indices].unsqueeze(1)

    # Compute PDE residual
    residual = pde_func(x, t, u, u_t_frac, u_xx)

    # Mean squared residual
    return torch.mean(residual ** 2)


def fractional_diffusion_loss(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    alpha: float,
    D: float,
    dt: Optional[float] = None,
) -> torch.Tensor:
    """
    Loss for fractional diffusion equation:
        D^α_t u = D (-Δ)^{β/2} u

    Specialized implementation for efficiency.

    Args:
        model: Neural network
        x: Spatial points
        t: Time points
        alpha: Temporal fractional order
        D: Diffusion coefficient
        dt: Time step

    Returns:
        PDE residual loss
    """
    def pde(x, t, u, u_t_frac, u_xx):
        # Laplacian
        laplacian = torch.sum(u_xx, dim=1, keepdim=True)

        # Residual: D^α_t u - D ∇²u
        return u_t_frac - D * laplacian

    return fractional_pde_loss(model, x, t, alpha, pde, dt)


def boundary_loss(
    model: nn.Module,
    x_boundary: torch.Tensor,
    t_boundary: torch.Tensor,
    bc_values: torch.Tensor,
    bc_type: str = "dirichlet"
) -> torch.Tensor:
    """
    Loss for boundary conditions.

    Args:
        model: Neural network
        x_boundary: Boundary points, shape (N, d)
        t_boundary: Time at boundary, shape (N, 1)
        bc_values: Boundary values
        bc_type: "dirichlet" (u=g) or "neumann" (∂u/∂n=g)

    Returns:
        Boundary condition loss
    """
    xt_boundary = torch.cat([x_boundary, t_boundary], dim=1)
    xt_boundary.requires_grad_(True)

    u_boundary = model(xt_boundary)

    if bc_type == "dirichlet":
        # u = g at boundary
        loss = torch.mean((u_boundary - bc_values) ** 2)

    elif bc_type == "neumann":
        # ∂u/∂n = g at boundary
        # Compute normal derivative (simplified for axis-aligned boundaries)
        grad_u = torch.autograd.grad(
            u_boundary, xt_boundary,
            grad_outputs=torch.ones_like(u_boundary),
            create_graph=True
        )[0]

        # Assume normal is along first spatial dimension (extend as needed)
        u_n = grad_u[:, 0:1]
        loss = torch.mean((u_n - bc_values) ** 2)

    else:
        raise ValueError(f"Unknown BC type: {bc_type}")

    return loss


def initial_condition_loss(
    model: nn.Module,
    x_initial: torch.Tensor,
    u_initial: torch.Tensor,
    t0: float = 0.0
) -> torch.Tensor:
    """
    Loss for initial condition: u(x, t0) = u0(x)

    Args:
        model: Neural network
        x_initial: Initial spatial points, shape (N, d)
        u_initial: Initial values, shape (N, 1)
        t0: Initial time

    Returns:
        Initial condition loss
    """
    t_initial = torch.full((x_initial.shape[0], 1), t0, device=x_initial.device)
    xt_initial = torch.cat([x_initial, t_initial], dim=1)

    u_pred = model(xt_initial)

    return torch.mean((u_pred - u_initial) ** 2)


def combined_pinn_loss(
    model: nn.Module,
    data_points: Dict[str, torch.Tensor],
    pde_points: Dict[str, torch.Tensor],
    bc_points: Dict[str, torch.Tensor],
    ic_points: Dict[str, torch.Tensor],
    pde_func: Callable,
    alpha: float,
    weights: Dict[str, float] = None,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Combined PINN loss with multiple components.

    Args:
        model: Neural network
        data_points: {"x": x_data, "t": t_data, "u": u_data}
        pde_points: {"x": x_pde, "t": t_pde}
        bc_points: {"x": x_bc, "t": t_bc, "u": u_bc, "type": bc_type}
        ic_points: {"x": x_ic, "u": u_ic, "t0": t0}
        pde_func: PDE residual function
        alpha: Fractional order
        weights: {"data": λ_data, "pde": λ_pde, "bc": λ_bc, "ic": λ_ic}

    Returns:
        (total_loss, loss_dict)
    """
    if weights is None:
        weights = {"data": 1.0, "pde": 1.0, "bc": 1.0, "ic": 1.0}

    loss_dict = {}

    # Data loss
    if data_points["x"] is not None:
        xt_data = torch.cat([data_points["x"], data_points["t"]], dim=1)
        u_pred = model(xt_data)
        loss_dict["data"] = data_loss(u_pred, data_points["u"])
    else:
        loss_dict["data"] = torch.tensor(0.0)

    # PDE loss
    if pde_points["x"] is not None:
        loss_dict["pde"] = fractional_pde_loss(
            model,
            pde_points["x"],
            pde_points["t"],
            alpha,
            pde_func
        )
    else:
        loss_dict["pde"] = torch.tensor(0.0)

    # Boundary loss
    if bc_points.get("x") is not None:
        loss_dict["bc"] = boundary_loss(
            model,
            bc_points["x"],
            bc_points["t"],
            bc_points["u"],
            bc_type=bc_points.get("type", "dirichlet")
        )
    else:
        loss_dict["bc"] = torch.tensor(0.0)

    # Initial condition loss
    if ic_points.get("x") is not None:
        loss_dict["ic"] = initial_condition_loss(
            model,
            ic_points["x"],
            ic_points["u"],
            t0=ic_points.get("t0", 0.0)
        )
    else:
        loss_dict["ic"] = torch.tensor(0.0)

    # Total loss
    total_loss = (
        weights["data"] * loss_dict["data"] +
        weights["pde"] * loss_dict["pde"] +
        weights["bc"] * loss_dict["bc"] +
        weights["ic"] * loss_dict["ic"]
    )

    # Convert to floats for logging
    loss_dict_float = {k: v.item() if isinstance(v, torch.Tensor) else v
                       for k, v in loss_dict.items()}

    return total_loss, loss_dict_float


def fractional_relaxation_loss(
    model: nn.Module,
    t: torch.Tensor,
    alpha: float,
    tau: float,
    y0: float = 1.0,
) -> torch.Tensor:
    """
    Loss for fractional relaxation equation:
        D^α y(t) = -y(t)/τ^α,  y(0) = y₀

    Useful for parameter estimation in BOLD modeling.

    Args:
        model: Neural network y(t)
        t: Time points
        alpha: Fractional order
        tau: Time constant
        y0: Initial value

    Returns:
        PDE + IC loss
    """
    t = t.reshape(-1, 1)
    t.requires_grad_(True)

    y = model(t)

    # Fractional derivative
    t_sorted, sort_indices = torch.sort(t.squeeze())
    y_sorted = y[sort_indices]
    dt = torch.mean(t_sorted[1:] - t_sorted[:-1]).item()

    dy_frac_sorted = caputo_derivative_autograd(y_sorted.squeeze(), alpha, dt)
    inverse_indices = torch.argsort(sort_indices)
    dy_frac = dy_frac_sorted[inverse_indices].unsqueeze(1)

    # PDE residual
    residual = dy_frac + y / (tau ** alpha)
    loss_pde = torch.mean(residual ** 2)

    # Initial condition
    y_0 = model(torch.zeros(1, 1, device=t.device))
    loss_ic = (y_0 - y0) ** 2

    return loss_pde + loss_ic


def causal_loss(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    window_size: int = 10
) -> torch.Tensor:
    """
    Causal consistency loss: predictions shouldn't depend on future.

    For time-series problems, ensures causality.

    Args:
        model: Neural network
        x: Spatial points
        t: Time points
        window_size: Temporal window for checking causality

    Returns:
        Causality violation penalty
    """
    # Sort by time
    t_sorted, sort_indices = torch.sort(t.squeeze())
    x_sorted = x[sort_indices]

    # Predictions at different times
    xt = torch.cat([x_sorted, t_sorted.unsqueeze(1)], dim=1)
    u_pred = model(xt)

    # Check if predictions at t depend on t+Δt
    # (approximate via gradient)
    t_sorted.requires_grad_(True)
    u_t = torch.autograd.grad(
        u_pred, t_sorted,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True
    )[0]

    # Penalize large future dependence (heuristic)
    # In practice, this is complex and problem-dependent
    causality_penalty = torch.mean(torch.abs(u_t))

    return causality_penalty


def regularization_loss(
    model: nn.Module,
    lambda_l2: float = 0.0,
    lambda_l1: float = 0.0
) -> torch.Tensor:
    """
    Regularization: L2 (weight decay) and L1 (sparsity).

    Args:
        model: Neural network
        lambda_l2: L2 regularization strength
        lambda_l1: L1 regularization strength

    Returns:
        Regularization loss
    """
    l2_reg = torch.tensor(0.0)
    l1_reg = torch.tensor(0.0)

    for param in model.parameters():
        l2_reg += torch.sum(param ** 2)
        l1_reg += torch.sum(torch.abs(param))

    return lambda_l2 * l2_reg + lambda_l1 * l1_reg
