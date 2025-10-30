"""
Validation tests for fractional derivatives using analytical solutions.

Tests verify numerical implementations against known exact results:
1. Power functions: D^α(t^p) = Γ(p+1)/Γ(p-α+1) * t^{p-α}
2. Exponential functions
3. Mittag-Leffler functions (solutions to fractional ODEs)
4. Convergence rates (should be O(dt^{1-α}) for L1, O(dt^{2-α}) for L2)
"""

import pytest
import torch
import numpy as np
from scipy.special import gamma

import sys
sys.path.append('..')

from src.fractional.derivatives import (
    caputo_derivative,
    riemann_liouville_derivative,
    grunwald_letnikov_derivative,
    fractional_integral,
)
from src.fractional.utils import compute_error_norm, estimate_convergence_rate


class TestCaputoDerivative:
    """Test suite for Caputo fractional derivative."""

    @pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7, 0.9])
    def test_power_function_t_squared(self, alpha):
        """
        Test D^α(t²) = 2 * Γ(3)/Γ(3-α) * t^{2-α}
        """
        t = torch.linspace(0.1, 10, 1000, dtype=torch.float64)
        dt = t[1] - t[0]

        # Numerical derivative
        f = t**2
        df_numerical = caputo_derivative(f, alpha, dt.item(), method="L1")

        # Analytical solution
        coeff = 2 * gamma(3) / gamma(3 - alpha)
        df_analytical = coeff * t**(2 - alpha)

        # Compute error (skip first few points due to initialization)
        error = compute_error_norm(
            df_numerical[10:],
            df_analytical[10:],
            norm_type="relative"
        )

        print(f"α={alpha:.1f}: Relative error = {error:.2e}")
        assert error < 0.05, f"Error too large for α={alpha}: {error}"

    @pytest.mark.parametrize("p,alpha", [
        (1.5, 0.5),
        (2.5, 0.5),
        (3.0, 0.5),
        (2.0, 0.3),
        (2.0, 0.7),
    ])
    def test_power_function_general(self, p, alpha):
        """
        Test D^α(t^p) = Γ(p+1)/Γ(p-α+1) * t^{p-α} for various p, α
        """
        if p <= alpha:
            pytest.skip(f"p={p} <= α={alpha}, derivative is zero/singular")

        t = torch.linspace(0.1, 5, 500, dtype=torch.float64)
        dt = t[1] - t[0]

        # Numerical
        f = t**p
        df_numerical = caputo_derivative(f, alpha, dt.item(), method="L1")

        # Analytical
        coeff = gamma(p + 1) / gamma(p - alpha + 1)
        df_analytical = coeff * t**(p - alpha)

        # Error
        error = compute_error_norm(
            df_numerical[10:],
            df_analytical[10:],
            norm_type="relative"
        )

        print(f"p={p}, α={alpha}: Error = {error:.2e}")
        assert error < 0.1, f"Error too large: {error}"

    def test_constant_function(self):
        """
        Test D^α(c) = 0 for any constant c.
        This is a key property of Caputo derivative.
        """
        t = torch.linspace(0, 10, 1000)
        dt = t[1] - t[0]

        f = 5.0 * torch.ones_like(t)
        alpha = 0.5

        df = caputo_derivative(f, alpha, dt.item())

        # Should be zero (up to numerical error)
        max_val = torch.max(torch.abs(df))
        print(f"Max value of D^{alpha}(constant): {max_val:.2e}")

        assert max_val < 1e-10, f"Derivative of constant should be zero, got {max_val}"

    def test_l1_vs_l2_accuracy(self):
        """
        Test that L2 scheme is more accurate than L1 scheme.
        """
        t = torch.linspace(0.1, 5, 200, dtype=torch.float64)
        dt = t[1] - t[0]
        alpha = 0.5

        f = t**2

        # Both methods
        df_l1 = caputo_derivative(f, alpha, dt.item(), method="L1")
        df_l2 = caputo_derivative(f, alpha, dt.item(), method="L2")

        # Analytical
        coeff = 2 * gamma(3) / gamma(3 - alpha)
        df_analytical = coeff * t**(2 - alpha)

        error_l1 = compute_error_norm(df_l1[10:], df_analytical[10:], "relative")
        error_l2 = compute_error_norm(df_l2[10:], df_analytical[10:], "relative")

        print(f"L1 error: {error_l1:.2e}")
        print(f"L2 error: {error_l2:.2e}")

        # L2 should be more accurate
        assert error_l2 < error_l1, "L2 scheme should be more accurate than L1"

    def test_convergence_rate_l1(self):
        """
        Test that L1 scheme has convergence order O(dt).
        """
        alpha = 0.5
        t_max = 2.0
        resolutions = [100, 200, 400, 800]

        errors = []
        dts = []

        for n in resolutions:
            t = torch.linspace(0.1, t_max, n, dtype=torch.float64)
            dt = t[1] - t[0]
            dts.append(dt.item())

            f = t**2
            df_numerical = caputo_derivative(f, alpha, dt.item(), method="L1")

            coeff = 2 * gamma(3) / gamma(3 - alpha)
            df_analytical = coeff * t**(2 - alpha)

            error = compute_error_norm(
                df_numerical[10:],
                df_analytical[10:],
                norm_type="L2"
            )
            errors.append(error)

        # Estimate convergence rate
        rate = estimate_convergence_rate(errors, dts)

        print(f"Estimated convergence rate: {rate:.2f}")
        print(f"Expected: ~1.0 for L1 scheme")

        # Should be close to 1 (first-order accuracy)
        assert 0.8 < rate < 1.3, f"Convergence rate {rate:.2f} outside expected range"


class TestRiemannLiouvilleDerivative:
    """Test suite for Riemann-Liouville derivative."""

    def test_constant_nonzero(self):
        """
        Key difference: RL derivative of constant ≠ 0.
        D^α(c) = c * t^{-α} / Γ(1-α)
        """
        t = torch.linspace(0.1, 5, 500, dtype=torch.float64)
        dt = t[1] - t[0]
        alpha = 0.5
        c = 1.0

        f = c * torch.ones_like(t)
        df = riemann_liouville_derivative(f, alpha, dt.item())

        # Analytical
        df_analytical = c * t**(-alpha) / gamma(1 - alpha)

        error = compute_error_norm(df[10:], df_analytical[10:], "relative")
        print(f"RL derivative of constant, error: {error:.2e}")

        assert error < 0.1, f"Error too large: {error}"

    def test_power_function(self):
        """
        Test RL derivative on power function.
        """
        t = torch.linspace(0.1, 5, 500, dtype=torch.float64)
        dt = t[1] - t[0]
        alpha = 0.5
        p = 2.0

        f = t**p
        df = riemann_liouville_derivative(f, alpha, dt.item())

        # Analytical: same as Caputo for smooth functions with f(0)=0
        coeff = gamma(p + 1) / gamma(p - alpha + 1)
        df_analytical = coeff * t**(p - alpha)

        error = compute_error_norm(df[20:], df_analytical[20:], "relative")
        print(f"RL power function error: {error:.2e}")

        assert error < 0.15


class TestGrunwaldLetnikovDerivative:
    """Test suite for Grünwald-Letnikov derivative."""

    def test_power_function(self):
        """
        Test GL derivative (converges to RL as dt→0).
        """
        t = torch.linspace(0.1, 5, 500, dtype=torch.float64)
        dt = t[1] - t[0]
        alpha = 0.5
        p = 2.0

        f = t**p
        df = grunwald_letnikov_derivative(f, alpha, dt.item())

        # Compare to analytical (RL formula)
        coeff = gamma(p + 1) / gamma(p - alpha + 1)
        df_analytical = coeff * t**(p - alpha)

        error = compute_error_norm(df[20:], df_analytical[20:], "relative")
        print(f"GL power function error: {error:.2e}")

        assert error < 0.15

    def test_memory_truncation(self):
        """
        Test that memory truncation doesn't significantly affect accuracy.
        """
        t = torch.linspace(0, 5, 500)
        dt = t[1] - t[0]
        alpha = 0.5

        f = t**2

        # Full memory
        df_full = grunwald_letnikov_derivative(f, alpha, dt.item(), max_memory=None)

        # Truncated memory
        df_trunc = grunwald_letnikov_derivative(f, alpha, dt.item(), max_memory=100)

        # Should be similar for most points
        error = compute_error_norm(df_full[100:], df_trunc[100:], "relative")
        print(f"Memory truncation error: {error:.2e}")

        assert error < 0.05


class TestFractionalIntegral:
    """Test suite for fractional integral."""

    def test_inverse_of_derivative(self):
        """
        Test that I^α D^α ≈ identity (approximately).
        """
        t = torch.linspace(0.1, 5, 500, dtype=torch.float64)
        dt = t[1] - t[0]
        alpha = 0.5

        # Original function
        f = t**2

        # Take derivative then integral
        df = caputo_derivative(f, alpha, dt.item())
        f_recovered = fractional_integral(df, alpha, dt.item())

        # Should recover original (up to constant and boundary effects)
        error = compute_error_norm(f[20:], f_recovered[20:], "relative")
        print(f"Inverse property error: {error:.2e}")

        assert error < 0.1

    def test_power_function(self):
        """
        Test I^α(t^p) = Γ(p+1)/Γ(p+α+1) * t^{p+α}
        """
        t = torch.linspace(0.1, 5, 500, dtype=torch.float64)
        dt = t[1] - t[0]
        alpha = 0.5
        p = 2.0

        f = t**p
        If = fractional_integral(f, alpha, dt.item())

        # Analytical
        coeff = gamma(p + 1) / gamma(p + alpha + 1)
        If_analytical = coeff * t**(p + alpha)

        error = compute_error_norm(If[10:], If_analytical[10:], "relative")
        print(f"Fractional integral error: {error:.2e}")

        assert error < 0.05


class TestMultidimensionalDerivatives:
    """Test fractional derivatives in multiple dimensions."""

    def test_2d_caputo(self):
        """
        Test 2D Caputo derivative.
        """
        from src.fractional.derivatives import caputo_derivative_2d

        nx, ny = 50, 50
        x = torch.linspace(0.1, 5, nx)
        y = torch.linspace(0.1, 5, ny)
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        X, Y = torch.meshgrid(x, y, indexing="ij")
        u = X**2 + Y**2

        alpha = 0.5

        du_dx, du_dy = caputo_derivative_2d(u, alpha, dx.item(), dy.item())

        # Analytical (approximate)
        coeff = gamma(3) / gamma(3 - alpha)
        du_dx_analytical = coeff * X**(2 - alpha)
        du_dy_analytical = coeff * Y**(2 - alpha)

        error_x = compute_error_norm(du_dx[5:, 5:], du_dx_analytical[5:, 5:], "relative")
        error_y = compute_error_norm(du_dy[5:, 5:], du_dy_analytical[5:, 5:], "relative")

        print(f"2D error x: {error_x:.2e}, y: {error_y:.2e}")

        assert error_x < 0.15 and error_y < 0.15


def test_mittag_leffler_solution():
    """
    Test that Mittag-Leffler function satisfies fractional ODE.

    D^α y(t) = -λ y(t), y(0) = 1
    Solution: y(t) = E_{α,1}(-λ t^α)
    """
    from src.fractional.special import mittag_leffler

    t = torch.linspace(0, 5, 200, dtype=torch.float64)
    dt = t[1] - t[0]
    alpha = 0.5
    lambda_ = 1.0

    # Solution
    y = torch.tensor([
        mittag_leffler(-lambda_ * t_val.item()**alpha, alpha, 1.0)
        for t_val in t
    ], dtype=torch.float64)

    # Check if it satisfies D^α y = -λ y
    dy = caputo_derivative(y, alpha, dt.item())
    rhs = -lambda_ * y

    # Error
    error = compute_error_norm(dy[10:], rhs[10:], "relative")
    print(f"Mittag-Leffler ODE error: {error:.2e}")

    # This is a challenging test, so allow larger error
    assert error < 0.2, f"Mittag-Leffler doesn't satisfy ODE: error = {error}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
