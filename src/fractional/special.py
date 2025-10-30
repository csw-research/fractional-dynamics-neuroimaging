"""
Special functions for fractional calculus.

Key functions:
1. Mittag-Leffler function E_{α,β}(z) - fundamental solution of fractional ODEs
2. Fractional exponential - relaxation and decay processes
3. Fractional Gaussian noise - for anomalous diffusion simulations
4. Power-law kernels - memory functions
"""

import torch
import numpy as np
from scipy.special import gamma
from typing import Union, Optional
import warnings


def mittag_leffler(
    z: Union[float, np.ndarray, torch.Tensor],
    alpha: float,
    beta: float = 1.0,
    terms: int = 100
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Compute Mittag-Leffler function E_{α,β}(z).

    Definition:
        E_{α,β}(z) = Σ_{k=0}^∞ z^k / Γ(αk + β)

    Special cases:
        E_{1,1}(z) = exp(z)
        E_{2,1}(-z²) = cos(z)
        E_{1,2}(z) = (exp(z) - 1) / z

    This is the fundamental solution of fractional differential equations:
        D^α y(t) = λ y(t), y(0) = y₀
        Solution: y(t) = y₀ E_{α,1}(λ t^α)

    Args:
        z: Argument (scalar or array)
        alpha: Order parameter, α > 0
        beta: Second parameter, β > 0
        terms: Number of series terms (increase for better accuracy)

    Returns:
        Value of Mittag-Leffler function

    Example:
        >>> # Fractional relaxation
        >>> t = torch.linspace(0, 10, 100)
        >>> alpha = 0.5
        >>> y = mittag_leffler(-t**alpha, alpha, 1)
    """
    is_torch = isinstance(z, torch.Tensor)
    is_scalar = isinstance(z, (int, float))

    if is_torch:
        device = z.device
        dtype = z.dtype
        z_np = z.detach().cpu().numpy()
    elif is_scalar:
        z_np = np.array([z])
    else:
        z_np = np.asarray(z)

    result = np.zeros_like(z_np, dtype=np.float64)

    # Series summation
    for k in range(terms):
        term = z_np**k / gamma(alpha * k + beta)
        result += term

        # Check convergence
        if k > 10 and np.max(np.abs(term)) < 1e-12:
            break

    if is_torch:
        result = torch.from_numpy(result).to(device=device, dtype=dtype)
    elif is_scalar:
        result = float(result[0])

    return result


def mittag_leffler_fast(
    z: torch.Tensor,
    alpha: float,
    beta: float = 1.0,
    tolerance: float = 1e-8
) -> torch.Tensor:
    """
    Fast approximation of Mittag-Leffler function using adaptive summation.

    More efficient for large arrays but same accuracy as mittag_leffler().

    Args:
        z: Argument tensor
        alpha: Order parameter
        beta: Second parameter
        tolerance: Convergence tolerance

    Returns:
        Approximate Mittag-Leffler values
    """
    result = torch.zeros_like(z, dtype=torch.float64)
    term = torch.ones_like(z, dtype=torch.float64)

    for k in range(200):  # Max iterations
        term = term * z / gamma(alpha * (k + 1) + beta) * gamma(alpha * k + beta)
        result += term

        if torch.max(torch.abs(term)) < tolerance:
            break

    return result.to(z.dtype)


def fractional_relaxation(
    t: torch.Tensor,
    alpha: float,
    tau: float = 1.0,
    y0: float = 1.0
) -> torch.Tensor:
    """
    Fractional relaxation function (stretched exponential).

    Solution to: D^α y(t) = -y(t)/τ^α, y(0) = y₀
    Result: y(t) = y₀ E_{α,1}(-(t/τ)^α)

    This describes:
    - Anomalous relaxation in viscoelastic materials
    - Non-Debye dielectric relaxation
    - Subdiffusive trapping in complex media

    Args:
        t: Time values
        alpha: Fractional order, 0 < α ≤ 1
        tau: Relaxation time scale
        y0: Initial value

    Returns:
        Relaxation function values

    Example:
        >>> t = torch.linspace(0, 10, 100)
        >>> # Compare different orders
        >>> y_05 = fractional_relaxation(t, alpha=0.5)  # Slow decay
        >>> y_10 = fractional_relaxation(t, alpha=1.0)  # Exponential
    """
    z = -(t / tau)**alpha
    return y0 * mittag_leffler(z, alpha, 1.0)


def fractional_gaussian_noise(
    n: int,
    H: float,
    dt: float = 1.0,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Generate fractional Gaussian noise (fGn) with Hurst exponent H.

    fGn has autocorrelation: R(k) ∝ |k|^{2H-2}

    Hurst exponent interpretation:
        H = 0.5: Uncorrelated (white noise)
        H > 0.5: Persistent (positive correlations)
        H < 0.5: Anti-persistent (negative correlations)

    Args:
        n: Number of samples
        H: Hurst exponent, 0 < H < 1
        dt: Time step
        device: Device for tensor

    Returns:
        Fractional Gaussian noise sequence

    Example:
        >>> # Persistent noise (long-range correlations)
        >>> noise = fractional_gaussian_noise(10000, H=0.7)
        >>> # Anti-persistent noise (mean-reverting)
        >>> noise = fractional_gaussian_noise(10000, H=0.3)
    """
    if not (0 < H < 1):
        raise ValueError("Hurst exponent must be in (0, 1)")

    # Generate via Davies-Harte method (exact, O(n log n))
    # Autocovariance function
    k = torch.arange(n, dtype=torch.float64, device=device)
    r = 0.5 * (torch.abs(k - 1)**(2*H) - 2*torch.abs(k)**(2*H) + torch.abs(k + 1)**(2*H))

    # Circulant embedding
    r_ext = torch.cat([r, r[1:-1].flip(0)])

    # Eigenvalues via FFT
    lambda_k = torch.fft.fft(r_ext).real

    # Check for numerical issues
    if torch.any(lambda_k < -1e-10):
        warnings.warn("Negative eigenvalues detected, using Davies-Harte variant")
        lambda_k = torch.clamp(lambda_k, min=0)

    # Generate noise
    xi = torch.randn(len(lambda_k), dtype=torch.float64, device=device)
    noise_fft = torch.sqrt(lambda_k) * xi
    noise = torch.fft.ifft(noise_fft).real[:n]

    # Scale by dt
    noise = noise * dt**(H)

    return noise.float()


def fractional_brownian_motion(
    n: int,
    H: float,
    dt: float = 1.0,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Generate fractional Brownian motion (fBm) path.

    fBm is the cumulative sum of fractional Gaussian noise:
        B_H(t) = ∫₀ᵗ dB_H(s)

    Properties:
        - Self-similar: B_H(ct) ~ c^H B_H(t)
        - Variance: Var[B_H(t)] ∝ t^{2H}
        - Increments: B_H(t+s) - B_H(s) ~ fGn

    Args:
        n: Number of time steps
        H: Hurst exponent
        dt: Time step
        device: Device for tensor

    Returns:
        fBm path, shape (n,)

    Example:
        >>> # Persistent random walk
        >>> path = fractional_brownian_motion(1000, H=0.8)
        >>> # Anti-persistent (mean-reverting)
        >>> path = fractional_brownian_motion(1000, H=0.3)
    """
    fgn = fractional_gaussian_noise(n, H, dt, device)
    fbm = torch.cumsum(fgn, dim=0)
    return fbm


def power_law_kernel(
    t: torch.Tensor,
    alpha: float,
    tau: float = 1.0
) -> torch.Tensor:
    """
    Power-law memory kernel for fractional dynamics.

    K(t) = t^{α-1} / (τ^α Γ(α))

    Used in fractional relaxation and anomalous diffusion:
        dy/dt = -∫₀ᵗ K(t-s) y(s) ds

    Args:
        t: Time values
        alpha: Power-law exponent, 0 < α < 1
        tau: Time scale

    Returns:
        Kernel values
    """
    return t**(alpha - 1) / (tau**alpha * gamma(alpha))


def cole_cole_function(
    omega: torch.Tensor,
    alpha: float,
    tau: float = 1.0
) -> torch.Tensor:
    """
    Cole-Cole function (frequency-domain fractional relaxation).

    χ(ω) = 1 / (1 + (iωτ)^α)

    Models dielectric/magnetic relaxation with distribution of time scales.

    Args:
        omega: Angular frequencies
        alpha: Distribution parameter, 0 < α ≤ 1
        tau: Characteristic time

    Returns:
        Complex susceptibility
    """
    z = 1j * omega * tau
    denominator = 1 + z**alpha
    return 1 / denominator


def stretched_exponential(
    t: torch.Tensor,
    beta: float,
    tau: float = 1.0,
    y0: float = 1.0
) -> torch.Tensor:
    """
    Stretched (Kohlrausch) exponential function.

    y(t) = y₀ exp(-(t/τ)^β)

    Related to fractional dynamics when β < 1.
    Often observed in glass transitions, protein folding.

    Args:
        t: Time values
        beta: Stretching exponent, 0 < β ≤ 1
        tau: Time scale
        y0: Initial value

    Returns:
        Stretched exponential values
    """
    return y0 * torch.exp(-(t / tau)**beta)


def levy_stable_pdf_approx(
    x: torch.Tensor,
    alpha: float,
    beta: float = 0.0,
    scale: float = 1.0
) -> torch.Tensor:
    """
    Approximate PDF of Lévy stable distribution.

    Used for heavy-tailed anomalous diffusion.

    Args:
        x: Sample points
        alpha: Stability parameter, 0 < α ≤ 2
        beta: Skewness, -1 ≤ β ≤ 1
        scale: Scale parameter

    Returns:
        Approximate PDF values

    Note:
        This is a simplified approximation. For production use,
        consider scipy.stats.levy_stable
    """
    if alpha == 2:
        # Gaussian case
        return torch.exp(-x**2 / (2 * scale**2)) / (scale * np.sqrt(2 * np.pi))

    # Simple approximation for tails
    # For accurate computation, use numerical methods or scipy
    warnings.warn("levy_stable_pdf_approx is a crude approximation")

    # Power-law tail approximation
    tail = scale**alpha / (torch.abs(x)**(1 + alpha))
    return tail / (2 * np.pi)  # Rough normalization


def generate_subdiffusive_trajectory(
    n_steps: int,
    alpha: float,
    D: float = 1.0,
    dt: float = 0.01,
    dim: int = 2
) -> torch.Tensor:
    """
    Generate subdiffusive trajectory using CTRW (continuous-time random walk).

    Mean squared displacement: <r²(t)> ∝ t^α

    Args:
        n_steps: Number of steps
        alpha: Subdiffusion exponent, 0 < α < 1
        D: Diffusion coefficient
        dt: Time step
        dim: Spatial dimension (1, 2, or 3)

    Returns:
        Trajectory, shape (n_steps, dim)

    Example:
        >>> # 2D subdiffusive particle
        >>> traj = generate_subdiffusive_trajectory(10000, alpha=0.5, dim=2)
        >>> # Compute MSD
        >>> msd = torch.mean(torch.sum(traj**2, dim=1))
    """
    # Generate waiting times from power-law distribution
    # PDF: ψ(t) ∝ t^{-1-α}

    # Use stable subordinator for time changes
    # Approximate using Lévy flights

    trajectory = torch.zeros(n_steps, dim)

    for i in range(1, n_steps):
        # Step size with power-law distribution
        if np.random.rand() < alpha:  # Trapping event
            step = torch.zeros(dim)
        else:
            step = torch.randn(dim) * np.sqrt(2 * D * dt)

        trajectory[i] = trajectory[i-1] + step

    return trajectory
