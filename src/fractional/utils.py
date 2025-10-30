"""
Utility functions for fractional calculus operations.
"""

import torch
import numpy as np
from typing import Union, Optional
from scipy.special import gamma


def check_fractional_order(alpha: float, valid_range: tuple = (0, 2)) -> None:
    """
    Validate fractional order parameter.

    Args:
        alpha: Fractional order to check
        valid_range: Tuple of (min, max) valid values

    Raises:
        ValueError: If alpha is outside valid range
    """
    min_val, max_val = valid_range
    if not (min_val < alpha <= max_val):
        raise ValueError(
            f"Fractional order α must be in ({min_val}, {max_val}], got {alpha}"
        )


def to_device(tensor: torch.Tensor, device: Optional[str] = None) -> torch.Tensor:
    """
    Move tensor to specified device.

    Args:
        tensor: Input tensor
        device: Target device ("cpu", "cuda", or None for auto-detect)

    Returns:
        Tensor on target device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return tensor.to(device)


def grunwald_weights(alpha: float, num_points: int) -> torch.Tensor:
    """
    Compute Grünwald-Letnikov weights for fractional derivative.

    The weights are computed recursively:
    w_0^{(α)} = 1
    w_k^{(α)} = (1 - (α+1)/k) * w_{k-1}^{(α)}

    Args:
        alpha: Fractional order
        num_points: Number of weights to compute

    Returns:
        Tensor of shape (num_points,) containing weights
    """
    weights = torch.zeros(num_points, dtype=torch.float64)
    weights[0] = 1.0

    for k in range(1, num_points):
        weights[k] = weights[k-1] * (1.0 - (alpha + 1.0) / k)

    return weights


def compute_error_norm(
    computed: torch.Tensor,
    analytical: torch.Tensor,
    norm_type: str = "L2"
) -> float:
    """
    Compute error between numerical and analytical solutions.

    Args:
        computed: Numerically computed solution
        analytical: Analytical solution
        norm_type: "L2", "Linf", or "relative"

    Returns:
        Error value
    """
    error = computed - analytical

    if norm_type == "L2":
        return torch.sqrt(torch.mean(error**2)).item()
    elif norm_type == "Linf":
        return torch.max(torch.abs(error)).item()
    elif norm_type == "relative":
        return (torch.norm(error) / torch.norm(analytical)).item()
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def estimate_convergence_rate(
    errors: list[float],
    grid_sizes: list[float]
) -> float:
    """
    Estimate convergence rate from errors at different grid sizes.

    Assumes power-law relationship: error ∝ h^p
    Returns p (convergence order).

    Args:
        errors: List of errors at different resolutions
        grid_sizes: Corresponding grid spacings

    Returns:
        Estimated convergence order
    """
    log_h = np.log(grid_sizes)
    log_e = np.log(errors)

    # Linear regression
    p = np.polyfit(log_h, log_e, 1)[0]

    return p


def generate_test_functions() -> dict:
    """
    Generate standard test functions for validation.

    Returns dictionary with:
        - "power": t^p for various p
        - "exponential": exp(λt)
        - "trigonometric": sin(ωt), cos(ωt)
        - "polynomial": sum of powers

    Returns:
        Dictionary of test functions and their properties
    """
    tests = {}

    # Power functions: t^p
    # Analytical: D^α(t^p) = Γ(p+1)/Γ(p-α+1) * t^{p-α}
    def power_func(p):
        return {
            "name": f"t^{p}",
            "func": lambda t: t**p,
            "derivative": lambda t, alpha: (
                gamma(p + 1) / gamma(p - alpha + 1) * t**(p - alpha)
                if p > alpha else torch.zeros_like(t)
            ),
            "order": p
        }

    tests["power_2"] = power_func(2.0)
    tests["power_1.5"] = power_func(1.5)
    tests["power_2.5"] = power_func(2.5)

    # Exponential: exp(λt)
    # Analytical: D^α(e^{λt}) = λ^α e^{λt} (approximate for small α)
    tests["exponential"] = {
        "name": "exp(t)",
        "func": lambda t: torch.exp(t),
        "derivative": lambda t, alpha: torch.exp(t) * (1.0)**(alpha),  # Approximate
    }

    # Mittag-Leffler function (exact solution to fractional ODE)
    tests["mittag_leffler"] = {
        "name": "E_{α,1}(-t^α)",
        "func": lambda t: torch.tensor([
            np.sum([(-t.item()**alpha)**k / gamma(alpha * k + 1)
                   for k in range(50)])
            for t in t
        ]),
    }

    return tests


def validate_autograd(
    func,
    input_tensor: torch.Tensor,
    epsilon: float = 1e-5
) -> bool:
    """
    Validate automatic differentiation using finite differences.

    Args:
        func: Function to test (should support autograd)
        input_tensor: Input requiring gradients
        epsilon: Finite difference step size

    Returns:
        True if gradients match finite differences (within tolerance)
    """
    input_tensor.requires_grad_(True)

    # Compute gradient via autograd
    output = func(input_tensor)
    output.sum().backward()
    grad_auto = input_tensor.grad.clone()

    # Compute gradient via finite differences
    input_tensor.requires_grad_(False)
    input_plus = input_tensor + epsilon
    input_minus = input_tensor - epsilon

    output_plus = func(input_plus)
    output_minus = func(input_minus)

    grad_fd = (output_plus - output_minus) / (2 * epsilon)

    # Compare
    rel_error = torch.norm(grad_auto - grad_fd) / torch.norm(grad_fd)

    return rel_error.item() < 1e-3


def memory_usage_mb() -> float:
    """
    Get current GPU memory usage in MB.

    Returns:
        Memory usage in megabytes, or 0 if CUDA not available
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def benchmark_function(
    func,
    *args,
    num_runs: int = 100,
    warmup: int = 10,
    **kwargs
) -> dict:
    """
    Benchmark function execution time.

    Args:
        func: Function to benchmark
        args: Function arguments
        num_runs: Number of timing runs
        warmup: Number of warmup runs
        kwargs: Function keyword arguments

    Returns:
        Dictionary with timing statistics
    """
    import time

    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timing
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
        "median_ms": np.median(times) * 1000,
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str
) -> None:
    """
    Save training checkpoint.

    Args:
        model: Neural network model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> dict:
    """
    Load training checkpoint.

    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state into

    Returns:
        Dictionary with checkpoint info
    """
    checkpoint = torch.load(filepath)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "epoch": checkpoint["epoch"],
        "loss": checkpoint["loss"],
    }


def create_spatial_grid(
    bounds: tuple,
    num_points: Union[int, tuple],
    dim: int = 2,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Create spatial grid for PDE solving.

    Args:
        bounds: ((x_min, x_max), (y_min, y_max), ...)
        num_points: Number of points (scalar or tuple for each dimension)
        dim: Spatial dimension
        device: Device for tensors

    Returns:
        Grid points, shape (N, dim)
    """
    if isinstance(num_points, int):
        num_points = (num_points,) * dim

    # Create 1D grids
    grids_1d = []
    for i, (n, (low, high)) in enumerate(zip(num_points, bounds)):
        grids_1d.append(torch.linspace(low, high, n, device=device))

    # Create meshgrid
    meshes = torch.meshgrid(*grids_1d, indexing="ij")

    # Flatten and stack
    grid = torch.stack([m.flatten() for m in meshes], dim=1)

    return grid


def sample_random_points(
    bounds: tuple,
    num_points: int,
    dim: int = 2,
    device: str = "cpu",
    method: str = "uniform"
) -> torch.Tensor:
    """
    Sample random points in domain.

    Args:
        bounds: ((x_min, x_max), (y_min, y_max), ...)
        num_points: Number of points to sample
        dim: Spatial dimension
        device: Device for tensors
        method: "uniform" or "latin_hypercube"

    Returns:
        Random points, shape (num_points, dim)
    """
    if method == "uniform":
        points = torch.rand(num_points, dim, device=device)

        # Scale to bounds
        for i, (low, high) in enumerate(bounds):
            points[:, i] = points[:, i] * (high - low) + low

        return points

    elif method == "latin_hypercube":
        # Latin hypercube sampling for better coverage
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=dim)
        points_np = sampler.random(n=num_points)

        points = torch.from_numpy(points_np).float().to(device)

        # Scale to bounds
        for i, (low, high) in enumerate(bounds):
            points[:, i] = points[:, i] * (high - low) + low

        return points

    else:
        raise ValueError(f"Unknown sampling method: {method}")
