"""
Fractional Laplacian operators for modeling anomalous spatial diffusion.

The fractional Laplacian (-Δ)^{s} with s ∈ (0, 1) is defined via:

1. Fourier transform: F[(-Δ)^{s} u](ξ) = |ξ|^{2s} F[u](ξ)
2. Integral formulation: (-Δ)^{s} u(x) = C(n,s) P.V. ∫ [u(x) - u(y)]/|x-y|^{n+2s} dy

where C(n,s) = 2^{2s} Γ((n+2s)/2) / (π^{n/2} Γ(-s))

The FFT-based method is most efficient and is used here.

Applications:
- Anomalous diffusion in white matter
- Non-local interactions in neural fields
- Lévy flights and heavy-tailed processes
"""

import torch
import numpy as np
from typing import Union, Optional


def fractional_laplacian_fft(
    u: torch.Tensor,
    alpha: float,
    dx: Union[float, tuple[float, ...]] = 1.0,
    boundary: str = "periodic"
) -> torch.Tensor:
    """
    Compute fractional Laplacian using FFT method.

    This computes (-Δ)^{α/2} u where α ∈ (0, 2).
    - α = 2: Classical Laplacian
    - α = 1: Square root of Laplacian
    - α < 1: Hypo-elliptic (subdiffusion)
    - α > 1: Hyper-elliptic

    Args:
        u: Input field, shape (Ny, Nx) or (Nz, Ny, Nx)
        alpha: Fractional power, 0 < α ≤ 2
        dx: Grid spacing (scalar or tuple for each dimension)
        boundary: "periodic" or "zero" boundary conditions

    Returns:
        Fractional Laplacian of u, same shape as input

    Example:
        >>> # 2D fractional diffusion
        >>> u = torch.randn(256, 256)
        >>> lap_u = fractional_laplacian_fft(u, alpha=1.5, dx=0.1)
    """
    if not (0 < alpha <= 2):
        raise ValueError(f"Alpha must be in (0, 2], got {alpha}")

    # Handle grid spacing
    if isinstance(dx, (int, float)):
        dx = (dx,) * u.ndim
    dx = torch.tensor(dx, dtype=u.dtype, device=u.device)

    # Apply boundary conditions
    if boundary == "zero":
        # Extend with zeros for non-periodic BC
        padding = [(0, s) for s in u.shape]
        u = torch.nn.functional.pad(u, sum(padding[::-1], ()))

    # FFT to frequency space
    u_hat = torch.fft.fftn(u)

    # Compute frequency grids
    freq_grids = []
    for i, (n, h) in enumerate(zip(u.shape, dx)):
        freq = torch.fft.fftfreq(n, d=h.item(), device=u.device)
        freq = 2 * np.pi * freq
        # Reshape for broadcasting
        shape = [1] * u.ndim
        shape[i] = n
        freq_grids.append(freq.reshape(shape))

    # Compute |ξ|^2
    xi_squared = sum(freq**2 for freq in freq_grids)

    # Fractional power: |ξ|^α
    with torch.no_grad():
        # Avoid 0^α at zero frequency
        xi_squared[tuple([0] * u.ndim)] = 0

    multiplier = xi_squared**(alpha / 2)

    # Apply in frequency space
    result_hat = multiplier * u_hat

    # Inverse FFT
    result = torch.fft.ifftn(result_hat).real

    # Remove padding if added
    if boundary == "zero":
        slices = tuple(slice(0, s) for s in u.shape)
        result = result[slices]

    return result


def fractional_laplacian(
    u: torch.Tensor,
    alpha: float,
    dx: float = 1.0,
    method: str = "fft"
) -> torch.Tensor:
    """
    Compute fractional Laplacian with multiple methods.

    Args:
        u: Input field
        alpha: Fractional power
        dx: Grid spacing
        method: "fft" (fast) or "matrix" (direct, for small problems)

    Returns:
        Fractional Laplacian
    """
    if method == "fft":
        return fractional_laplacian_fft(u, alpha, dx)
    elif method == "matrix":
        return fractional_laplacian_matrix(u, alpha, dx)
    else:
        raise ValueError(f"Unknown method: {method}")


def fractional_laplacian_matrix(
    u: torch.Tensor,
    alpha: float,
    dx: float = 1.0
) -> torch.Tensor:
    """
    Compute fractional Laplacian using matrix method (direct).

    This constructs the full operator matrix and applies it.
    Only suitable for small problems (N < 1000) due to O(N²) memory.

    Args:
        u: 1D field, shape (N,)
        alpha: Fractional power
        dx: Grid spacing

    Returns:
        Fractional Laplacian
    """
    if u.ndim != 1:
        raise ValueError("Matrix method only supports 1D fields")

    n = u.shape[0]

    # Build fractional Laplacian matrix via eigendecomposition
    # For 1D with periodic BC: eigenfunctions are exp(i k x)
    # eigenvalues are -k²

    # Construct standard Laplacian matrix
    diag = -2 * torch.ones(n, device=u.device) / dx**2
    off_diag = torch.ones(n-1, device=u.device) / dx**2

    L = torch.diag(diag) + torch.diag(off_diag, 1) + torch.diag(off_diag, -1)

    # Periodic BC
    L[0, -1] = 1 / dx**2
    L[-1, 0] = 1 / dx**2

    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(L)

    # Fractional power: Λ^{α/2}
    # Clip negative eigenvalues (numerical artifacts)
    eigenvalues = torch.abs(eigenvalues)
    eigenvalues_frac = eigenvalues**(alpha / 2)

    # Reconstruct: L^{α/2} = V Λ^{α/2} V^T
    L_frac = eigenvectors @ torch.diag(eigenvalues_frac) @ eigenvectors.T

    return L_frac @ u


def riesz_potential(
    u: torch.Tensor,
    s: float,
    dx: Union[float, tuple[float, ...]] = 1.0
) -> torch.Tensor:
    """
    Compute Riesz potential (inverse of fractional Laplacian).

    I_s = (-Δ)^{-s/2}

    This is the fractional integral operator in multiple dimensions.

    Args:
        u: Input field
        s: Order of potential, s > 0
        dx: Grid spacing

    Returns:
        Riesz potential of u
    """
    return fractional_laplacian_fft(u, -s, dx)


def tempered_fractional_laplacian(
    u: torch.Tensor,
    alpha: float,
    lambda_: float,
    dx: Union[float, tuple[float, ...]] = 1.0
) -> torch.Tensor:
    """
    Compute tempered fractional Laplacian.

    The tempered operator includes exponential damping:
    (-Δ + λ²)^{α/2} u

    This is useful for modeling bounded processes where far-field
    interactions decay exponentially.

    Args:
        u: Input field
        alpha: Fractional power
        lambda_: Tempering parameter (decay rate)
        dx: Grid spacing

    Returns:
        Tempered fractional Laplacian

    Reference:
        Meerschaert & Sikorskii (2012), Stochastic Models for
        Fractional Calculus
    """
    if isinstance(dx, (int, float)):
        dx = (dx,) * u.ndim
    dx = torch.tensor(dx, dtype=u.dtype, device=u.device)

    # FFT to frequency space
    u_hat = torch.fft.fftn(u)

    # Compute frequency grids
    freq_grids = []
    for i, (n, h) in enumerate(zip(u.shape, dx)):
        freq = torch.fft.fftfreq(n, d=h.item(), device=u.device)
        freq = 2 * np.pi * freq
        shape = [1] * u.ndim
        shape[i] = n
        freq_grids.append(freq.reshape(shape))

    # Compute |ξ|² + λ²
    xi_squared = sum(freq**2 for freq in freq_grids) + lambda_**2

    # Fractional power: (|ξ|² + λ²)^{α/2}
    multiplier = xi_squared**(alpha / 2)

    # Apply in frequency space
    result_hat = multiplier * u_hat

    # Inverse FFT
    result = torch.fft.ifftn(result_hat).real

    return result


def anisotropic_fractional_laplacian(
    u: torch.Tensor,
    alpha: Union[float, tuple[float, ...]],
    dx: Union[float, tuple[float, ...]] = 1.0
) -> torch.Tensor:
    """
    Compute anisotropic fractional Laplacian with different orders
    along each axis.

    Useful for white matter where diffusion is highly directional.

    Args:
        u: Input field
        alpha: Fractional power (scalar or tuple for each dimension)
        dx: Grid spacing

    Returns:
        Anisotropic fractional Laplacian

    Example:
        >>> # Stronger diffusion in x than y
        >>> u = torch.randn(256, 256)
        >>> lap_u = anisotropic_fractional_laplacian(u, alpha=(1.8, 1.2))
    """
    if isinstance(alpha, (int, float)):
        alpha = (alpha,) * u.ndim
    if isinstance(dx, (int, float)):
        dx = (dx,) * u.ndim

    if len(alpha) != u.ndim or len(dx) != u.ndim:
        raise ValueError("alpha and dx must match number of dimensions")

    # FFT to frequency space
    u_hat = torch.fft.fftn(u)

    # Compute frequency grids and multiplier
    multiplier = 1.0
    for i, (n, h, a) in enumerate(zip(u.shape, dx, alpha)):
        freq = torch.fft.fftfreq(n, d=h, device=u.device)
        freq = 2 * np.pi * freq
        shape = [1] * u.ndim
        shape[i] = n
        freq = freq.reshape(shape)

        multiplier = multiplier * (torch.abs(freq)**a)

    # Apply in frequency space
    result_hat = multiplier * u_hat

    # Inverse FFT
    result = torch.fft.ifftn(result_hat).real

    return result


def fractional_diffusion_step(
    u: torch.Tensor,
    alpha: float,
    D: float,
    dt: float,
    dx: Union[float, tuple[float, ...]],
    method: str = "explicit"
) -> torch.Tensor:
    """
    Single time step for fractional diffusion equation:
    ∂u/∂t = D (-Δ)^{α/2} u

    Args:
        u: Current field
        alpha: Fractional diffusion exponent
        D: Diffusion coefficient
        dt: Time step
        dx: Spatial grid spacing
        method: "explicit" or "implicit"

    Returns:
        Field at next time step

    Note:
        Explicit method: u^{n+1} = u^n + dt * D * (-Δ)^{α/2} u^n
        For stability, require dt < C * dx^α where C depends on D, α
    """
    if method == "explicit":
        lap_u = fractional_laplacian_fft(u, alpha, dx)
        u_next = u + dt * D * lap_u
        return u_next

    elif method == "implicit":
        # Solve (I - dt * D * L^α) u^{n+1} = u^n
        # Use fixed-point iteration or conjugate gradient
        raise NotImplementedError("Implicit method not yet implemented")

    else:
        raise ValueError(f"Unknown method: {method}")
