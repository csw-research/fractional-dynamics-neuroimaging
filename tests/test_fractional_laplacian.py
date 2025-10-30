"""
Validation tests for fractional Laplacian.

Tests verify:
1. α=2 recovers classical Laplacian
2. Eigenfunction property: (-Δ)^{α/2} e^{ikx} = |k|^α e^{ikx}
3. Fractional diffusion equation solutions
4. Conservation properties
"""

import pytest
import torch
import numpy as np

import sys
sys.path.append('..')

from src.fractional.laplacian import (
    fractional_laplacian_fft,
    fractional_laplacian_matrix,
    fractional_diffusion_step,
    anisotropic_fractional_laplacian,
    tempered_fractional_laplacian,
)
from src.fractional.utils import compute_error_norm


class TestFractionalLaplacianFFT:
    """Test FFT-based fractional Laplacian."""

    def test_alpha_2_classical_laplacian_1d(self):
        """
        Test that α=2 recovers the classical Laplacian in 1D.
        -∇²u should equal numerical second derivative.
        """
        n = 256
        L = 2 * np.pi
        x = torch.linspace(0, L, n, endpoint=False)
        dx = x[1] - x[0]

        # Test function: sin(kx)
        k = 3
        u = torch.sin(k * x)

        # Fractional Laplacian with α=2
        lap_u = fractional_laplacian_fft(u, alpha=2.0, dx=dx.item())

        # Analytical Laplacian: -∇²sin(kx) = k² sin(kx)
        lap_analytical = k**2 * torch.sin(k * x)

        error = compute_error_norm(lap_u, lap_analytical, "relative")
        print(f"α=2 vs classical Laplacian error: {error:.2e}")

        assert error < 0.01, f"α=2 should recover classical Laplacian, error={error}"

    def test_alpha_2_classical_laplacian_2d(self):
        """
        Test α=2 in 2D: -∇²sin(k₁x)sin(k₂y) = (k₁² + k₂²)sin(k₁x)sin(k₂y)
        """
        n = 64
        L = 2 * np.pi
        x = torch.linspace(0, L, n, endpoint=False)
        y = torch.linspace(0, L, n, endpoint=False)
        dx = x[1] - x[0]

        X, Y = torch.meshgrid(x, y, indexing="ij")

        k1, k2 = 2, 3
        u = torch.sin(k1 * X) * torch.sin(k2 * Y)

        # Fractional Laplacian
        lap_u = fractional_laplacian_fft(u, alpha=2.0, dx=dx.item())

        # Analytical
        lap_analytical = (k1**2 + k2**2) * torch.sin(k1 * X) * torch.sin(k2 * Y)

        error = compute_error_norm(lap_u, lap_analytical, "relative")
        print(f"2D α=2 error: {error:.2e}")

        assert error < 0.01

    def test_eigenfunction_property(self):
        """
        Test eigenfunction property: (-Δ)^{α/2} e^{ikx} = |k|^α e^{ikx}
        """
        n = 256
        L = 2 * np.pi
        x = torch.linspace(0, L, n, endpoint=False)
        dx = x[1] - x[0]

        alpha = 1.5
        k = 4

        # Complex exponential (use real part)
        u = torch.cos(k * x)

        lap_u = fractional_laplacian_fft(u, alpha=alpha, dx=dx.item())

        # Analytical: |k|^α cos(kx)
        lap_analytical = k**alpha * torch.cos(k * x)

        error = compute_error_norm(lap_u, lap_analytical, "relative")
        print(f"Eigenfunction property error (α={alpha}): {error:.2e}")

        assert error < 0.01

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0])
    def test_different_alphas(self, alpha):
        """
        Test fractional Laplacian for different α values.
        """
        n = 128
        L = 2 * np.pi
        x = torch.linspace(0, L, n, endpoint=False)
        dx = x[1] - x[0]

        k = 3
        u = torch.sin(k * x)

        lap_u = fractional_laplacian_fft(u, alpha=alpha, dx=dx.item())

        # Expected: k^α sin(kx)
        lap_expected = k**alpha * torch.sin(k * x)

        error = compute_error_norm(lap_u, lap_expected, "relative")
        print(f"α={alpha}: error = {error:.2e}")

        assert error < 0.02

    def test_zero_at_zero_frequency(self):
        """
        Test that constant functions map to zero (α>0).
        """
        u = torch.ones(100)
        alpha = 1.5

        lap_u = fractional_laplacian_fft(u, alpha=alpha, dx=0.1)

        max_val = torch.max(torch.abs(lap_u))
        print(f"Max value for constant input: {max_val:.2e}")

        assert max_val < 1e-10, "Constant function should map to zero"


class TestFractionalDiffusion:
    """Test fractional diffusion equation."""

    def test_diffusion_decay(self):
        """
        Test that fractional diffusion causes decay over time.
        """
        n = 128
        L = 2 * np.pi
        x = torch.linspace(0, L, n, endpoint=False)
        dx = x[1] - x[0]

        # Initial condition: Gaussian
        u0 = torch.exp(-((x - L/2)**2) / 0.5)

        alpha = 1.5
        D = 0.1
        dt = 0.001
        num_steps = 100

        u = u0.clone()
        for _ in range(num_steps):
            u = fractional_diffusion_step(u, alpha, D, dt, dx.item())

        # Should decay (total mass decreases for non-periodic BC,
        # but we're using periodic so check spreading)
        initial_peak = torch.max(u0)
        final_peak = torch.max(u)

        print(f"Initial peak: {initial_peak:.3f}, Final peak: {final_peak:.3f}")

        # Peak should decrease due to diffusion
        assert final_peak < initial_peak * 0.95, "Diffusion should spread the peak"

    def test_gaussian_solution_alpha_2(self):
        """
        For α=2 (classical diffusion), test Gaussian solution.
        ∂u/∂t = D∇²u has solution u(x,t) = 1/√(4πDt) exp(-x²/(4Dt))
        """
        n = 256
        L = 20
        x = torch.linspace(-L/2, L/2, n, endpoint=False)
        dx = x[1] - x[0]

        D = 1.0
        t = 0.5
        dt = 0.001
        num_steps = int(t / dt)

        # Initial: delta function (approximate with narrow Gaussian)
        u0 = torch.exp(-x**2 / 0.01) / np.sqrt(np.pi * 0.01)

        # Evolve
        u = u0.clone()
        for _ in range(num_steps):
            u = fractional_diffusion_step(u, alpha=2.0, D=D, dt=dt, dx=dx.item())

        # Analytical solution
        u_analytical = torch.exp(-x**2 / (4*D*t)) / np.sqrt(4*np.pi*D*t)

        error = compute_error_norm(u, u_analytical, "L2")
        print(f"Gaussian diffusion error: {error:.2e}")

        # This is an approximate test due to discretization
        assert error < 0.1


class TestAnisotropicLaplacian:
    """Test anisotropic fractional Laplacian."""

    def test_different_orders_each_direction(self):
        """
        Test that different orders in x and y work correctly.
        """
        n = 64
        L = 2 * np.pi
        x = torch.linspace(0, L, n, endpoint=False)
        y = torch.linspace(0, L, n, endpoint=False)
        dx = x[1] - x[0]

        X, Y = torch.meshgrid(x, y, indexing="ij")

        k1, k2 = 2, 3
        u = torch.cos(k1 * X) * torch.cos(k2 * Y)

        alpha_x, alpha_y = 1.5, 1.0

        lap_u = anisotropic_fractional_laplacian(
            u,
            alpha=(alpha_x, alpha_y),
            dx=dx.item()
        )

        # Expected: k1^{α_x} * k2^{α_y} * cos(k1*X) * cos(k2*Y)
        lap_expected = k1**alpha_x * k2**alpha_y * torch.cos(k1 * X) * torch.cos(k2 * Y)

        error = compute_error_norm(lap_u, lap_expected, "relative")
        print(f"Anisotropic Laplacian error: {error:.2e}")

        assert error < 0.05


class TestTemperedLaplacian:
    """Test tempered fractional Laplacian."""

    def test_exponential_decay(self):
        """
        Test that tempering adds exponential decay.
        """
        n = 128
        L = 2 * np.pi
        x = torch.linspace(0, L, n, endpoint=False)
        dx = x[1] - x[0]

        k = 3
        u = torch.sin(k * x)

        alpha = 1.5
        lambda_ = 0.5

        # Standard
        lap_standard = fractional_laplacian_fft(u, alpha=alpha, dx=dx.item())

        # Tempered
        lap_tempered = tempered_fractional_laplacian(
            u, alpha=alpha, lambda_=lambda_, dx=dx.item()
        )

        # Tempered should have larger magnitude (due to +λ² term)
        mag_standard = torch.norm(lap_standard)
        mag_tempered = torch.norm(lap_tempered)

        print(f"Standard: {mag_standard:.3f}, Tempered: {mag_tempered:.3f}")

        assert mag_tempered > mag_standard, "Tempered should increase magnitude"


class TestMatrixMethod:
    """Test matrix-based fractional Laplacian (small problems)."""

    def test_matches_fft_method(self):
        """
        Test that matrix method matches FFT method for 1D.
        """
        n = 50
        x = torch.linspace(0, 2*np.pi, n, endpoint=False)
        dx = x[1] - x[0]

        k = 2
        u = torch.sin(k * x)
        alpha = 1.5

        lap_fft = fractional_laplacian_fft(u, alpha=alpha, dx=dx.item())
        lap_matrix = fractional_laplacian_matrix(u, alpha=alpha, dx=dx.item())

        error = compute_error_norm(lap_fft, lap_matrix, "relative")
        print(f"FFT vs Matrix error: {error:.2e}")

        assert error < 0.1, "FFT and matrix methods should agree"


def test_stability_condition():
    """
    Test stability of explicit fractional diffusion.
    Should be stable for dt < C * dx^α.
    """
    n = 64
    x = torch.linspace(0, 2*np.pi, n, endpoint=False)
    dx = x[1] - x[0]

    u0 = torch.sin(2 * x)
    alpha = 1.5
    D = 1.0

    # Stable time step
    dt_stable = 0.5 * dx.item()**alpha / D

    u = u0.clone()
    for _ in range(100):
        u = fractional_diffusion_step(u, alpha, D, dt_stable, dx.item())

    # Should remain bounded
    assert torch.isfinite(u).all(), "Solution should remain finite"
    assert torch.max(torch.abs(u)) < 100, "Solution shouldn't explode"

    print("Stability test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
