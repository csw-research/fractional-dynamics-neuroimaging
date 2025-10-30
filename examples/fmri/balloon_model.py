"""
Fractional balloon model for BOLD signal dynamics.

Extends the classical Buxton-Friston balloon model with fractional derivatives
to capture memory effects in neurovascular coupling.

Classical model:
    ds/dt = ε u(t) - κ s(t) - γ(f(t) - 1)
    df/dt = s(t)
    dv/dt = (f(t) - v(t)^{1/α}) / τ
    dq/dt = (f(t) E(f,E₀) - v(t)^{1/α} q(t)/v(t)) / τ

Fractional extension:
    D^α s(t) = ε u(t) - κ s(t) - γ(f(t) - 1)
    D^β f(t) = s(t)

where α, β ∈ (0, 1] capture memory in the hemodynamic response.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
from scipy.integrate import odeint
from scipy.optimize import minimize

import sys
sys.path.append('../..')

from src.fractional.derivatives import caputo_derivative
from src.fractional.special import mittag_leffler, fractional_relaxation
from src.pinns.models import FractionalPINN
from src.pinns.losses import fractional_relaxation_loss
from src.pinns.training import train_pinn


class ClassicalBalloonModel:
    """
    Classical (integer-order) balloon model.

    Reference:
        Buxton et al. (1998), Friston et al. (2000)
    """

    def __init__(
        self,
        epsilon: float = 0.5,
        kappa: float = 0.65,
        gamma: float = 0.41,
        tau: float = 2.0,
        alpha: float = 0.33,
        E0: float = 0.4,
        V0: float = 0.03,
    ):
        """
        Initialize balloon model parameters.

        Args:
            epsilon: Neuronal efficacy
            kappa: Signal decay
            gamma: Autoregulation
            tau: Transit time
            alpha: Vessel stiffness
            E0: Resting oxygen extraction
            V0: Resting blood volume fraction
        """
        self.epsilon = epsilon
        self.kappa = kappa
        self.gamma = gamma
        self.tau = tau
        self.alpha_vessel = alpha  # Rename to avoid confusion with fractional order
        self.E0 = E0
        self.V0 = V0

    def _ode_system(
        self,
        state: np.ndarray,
        t: float,
        u: callable
    ) -> np.ndarray:
        """
        ODE system for balloon model.

        State: [s, f, v, q]
        """
        s, f, v, q = state

        # Oxygen extraction
        E = 1 - (1 - self.E0)**(1 / f) if f > 0 else self.E0

        # ODEs
        ds_dt = self.epsilon * u(t) - self.kappa * s - self.gamma * (f - 1)
        df_dt = s
        dv_dt = (f - v**(1 / self.alpha_vessel)) / self.tau
        dq_dt = (f * E - v**(1 / self.alpha_vessel) * q / v) / self.tau

        return np.array([ds_dt, df_dt, dv_dt, dq_dt])

    def simulate(
        self,
        neural_input: np.ndarray,
        t: np.ndarray,
        initial_state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate BOLD response.

        Args:
            neural_input: Neural activity u(t), shape (N,)
            t: Time points, shape (N,)
            initial_state: Initial [s, f, v, q] (default: equilibrium)

        Returns:
            (states, bold_signal): Full state trajectory and BOLD signal
        """
        if initial_state is None:
            initial_state = np.array([0, 1, 1, 1])

        # Interpolate neural input
        from scipy.interpolate import interp1d
        u_func = interp1d(t, neural_input, kind='linear', fill_value="extrapolate")

        # Solve ODE
        states = odeint(self._ode_system, initial_state, t, args=(u_func,))

        # BOLD signal (Balloon equation)
        s, f, v, q = states.T
        bold = self.V0 * (
            7 * self.E0 * (1 - q) +
            2 * (1 - q / v) +
            (2 * self.E0 - 0.2) * (1 - v)
        )

        return states, bold


class FractionalBalloonModel:
    """
    Fractional balloon model with memory effects.

    Replaces integer derivatives with fractional:
        D^α s, D^β f instead of ds/dt, df/dt
    """

    def __init__(
        self,
        alpha: float = 0.8,
        beta: float = 0.9,
        epsilon: float = 0.5,
        kappa: float = 0.65,
        gamma: float = 0.41,
        tau: float = 2.0,
        alpha_vessel: float = 0.33,
        E0: float = 0.4,
        V0: float = 0.03,
    ):
        """
        Initialize fractional balloon model.

        Args:
            alpha: Fractional order for signal equation
            beta: Fractional order for flow equation
            (other params same as classical model)
        """
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.kappa = kappa
        self.gamma = gamma
        self.tau = tau
        self.alpha_vessel = alpha_vessel
        self.E0 = E0
        self.V0 = V0

    def simulate(
        self,
        neural_input: np.ndarray,
        t: np.ndarray,
        dt: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate fractional BOLD response.

        Uses numerical scheme for fractional ODEs.
        """
        if dt is None:
            dt = t[1] - t[0]

        n = len(t)

        # Initialize state variables
        s = np.zeros(n)
        f = np.ones(n)
        v = np.ones(n)
        q = np.ones(n)

        u = neural_input

        # Numerical integration (Euler-like scheme for fractional ODEs)
        for i in range(1, n):
            # Compute fractional derivatives at previous time
            if i > 2:
                s_frac = caputo_derivative(
                    torch.from_numpy(s[:i]),
                    self.alpha,
                    dt,
                    method="L1"
                ).numpy()[-1]

                f_frac = caputo_derivative(
                    torch.from_numpy(f[:i]),
                    self.beta,
                    dt,
                    method="L1"
                ).numpy()[-1]
            else:
                s_frac = 0
                f_frac = 0

            # Right-hand side
            rhs_s = self.epsilon * u[i-1] - self.kappa * s[i-1] - self.gamma * (f[i-1] - 1)
            rhs_f = s[i-1]

            # Update (implicit-like scheme)
            s[i] = s[i-1] + dt * (rhs_s - s_frac)
            f[i] = f[i-1] + dt * (rhs_f - f_frac)

            # Classical equations for v, q (could also be fractionalized)
            E = 1 - (1 - self.E0)**(1 / f[i]) if f[i] > 0 else self.E0

            dv = (f[i] - v[i-1]**(1 / self.alpha_vessel)) / self.tau
            v[i] = v[i-1] + dt * dv

            dq = (f[i] * E - v[i]**(1 / self.alpha_vessel) * q[i-1] / v[i]) / self.tau
            q[i] = q[i-1] + dt * dq

        # BOLD signal
        bold = self.V0 * (
            7 * self.E0 * (1 - q) +
            2 * (1 - q / v) +
            (2 * self.E0 - 0.2) * (1 - v)
        )

        states = np.column_stack([s, f, v, q])

        return states, bold

    def fit(
        self,
        bold_observed: np.ndarray,
        neural_input: np.ndarray,
        t: np.ndarray,
        fix_alpha: bool = False,
        fix_beta: bool = False
    ) -> Dict[str, float]:
        """
        Fit fractional orders to observed BOLD data.

        Args:
            bold_observed: Measured BOLD signal
            neural_input: Neural input
            t: Time points
            fix_alpha: Whether to fix α (not optimize)
            fix_beta: Whether to fix β

        Returns:
            Dictionary of fitted parameters
        """
        initial_params = [self.alpha, self.beta, self.epsilon, self.kappa]

        def objective(params):
            alpha, beta, epsilon, kappa = params

            # Constraints
            if not (0 < alpha <= 1 and 0 < beta <= 1):
                return 1e10

            model = FractionalBalloonModel(
                alpha=alpha,
                beta=beta,
                epsilon=epsilon,
                kappa=kappa,
                gamma=self.gamma,
                tau=self.tau,
            )

            _, bold_sim = model.simulate(neural_input, t)

            # Mean squared error
            mse = np.mean((bold_observed - bold_sim)**2)

            return mse

        # Optimize
        bounds = [
            (0.1, 1.0) if not fix_alpha else (self.alpha, self.alpha),
            (0.1, 1.0) if not fix_beta else (self.beta, self.beta),
            (0.1, 1.0),  # epsilon
            (0.1, 1.0),  # kappa
        ]

        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100}
        )

        fitted_params = {
            'alpha': result.x[0],
            'beta': result.x[1],
            'epsilon': result.x[2],
            'kappa': result.x[3],
            'mse': result.fun,
        }

        return fitted_params


class PINNBalloonModel(nn.Module):
    """
    PINN-based fractional balloon model.

    Learns hemodynamic response function as neural network,
    constrained by fractional ODEs.
    """

    def __init__(
        self,
        hidden_layers: list = [32, 32, 32],
        alpha: float = 0.8,
        beta: float = 0.9,
    ):
        super().__init__()

        self.pinn_s = FractionalPINN(
            input_dim=1,  # t
            output_dim=1,  # s(t)
            hidden_layers=hidden_layers,
            fractional_order=alpha,
        )

        self.pinn_f = FractionalPINN(
            input_dim=1,
            output_dim=1,  # f(t)
            hidden_layers=hidden_layers,
            fractional_order=beta,
        )

        self.alpha = alpha
        self.beta = beta

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: predict s(t) and f(t).
        """
        s = self.pinn_s(t)
        f = self.pinn_f(t)

        return s, f


def compare_classical_vs_fractional(
    neural_input: np.ndarray,
    t: np.ndarray,
    alpha: float = 0.7,
    beta: float = 0.8,
    plot: bool = True
) -> Dict[str, np.ndarray]:
    """
    Compare classical vs fractional balloon models.

    Args:
        neural_input: Neural activity
        t: Time points
        alpha, beta: Fractional orders
        plot: Whether to create comparison plot

    Returns:
        Dictionary with results
    """
    # Classical model
    classical = ClassicalBalloonModel()
    states_classical, bold_classical = classical.simulate(neural_input, t)

    # Fractional model
    fractional = FractionalBalloonModel(alpha=alpha, beta=beta)
    states_fractional, bold_fractional = fractional.simulate(neural_input, t)

    results = {
        'classical_bold': bold_classical,
        'fractional_bold': bold_fractional,
        'classical_states': states_classical,
        'fractional_states': states_fractional,
    }

    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Neural input
        axes[0].plot(t, neural_input, 'k-', linewidth=2)
        axes[0].set_ylabel('Neural activity u(t)', fontsize=12)
        axes[0].set_title('Input and Hemodynamic Response', fontsize=14)
        axes[0].grid(True, alpha=0.3)

        # Flow
        axes[1].plot(t, states_classical[:, 1], 'b-', label='Classical (α=1)', linewidth=2)
        axes[1].plot(t, states_fractional[:, 1], 'r--', label=f'Fractional (β={beta})', linewidth=2)
        axes[1].set_ylabel('CBF f(t)', fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        # BOLD signal
        axes[2].plot(t, bold_classical, 'b-', label='Classical', linewidth=2)
        axes[2].plot(t, bold_fractional, 'r--', label='Fractional', linewidth=2)
        axes[2].set_xlabel('Time (s)', fontsize=12)
        axes[2].set_ylabel('BOLD signal (%)', fontsize=12)
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('balloon_model_comparison.png', dpi=150, bbox_inches='tight')
        print("Saved plot to: balloon_model_comparison.png")

    return results


if __name__ == "__main__":
    # Example: Compare classical vs fractional balloon model

    # Create neural input (block design)
    t = np.linspace(0, 60, 300)  # 60 seconds, TR=0.2s
    neural_input = np.zeros_like(t)
    neural_input[(t > 10) & (t < 20)] = 1.0  # Block 1
    neural_input[(t > 35) & (t < 45)] = 1.0  # Block 2

    # Add noise
    neural_input += 0.1 * np.random.randn(len(t))

    # Compare models
    print("Comparing classical vs fractional balloon models...")
    results = compare_classical_vs_fractional(
        neural_input,
        t,
        alpha=0.7,
        beta=0.8,
        plot=True
    )

    print("\nKey observations:")
    print("- Fractional model shows slower rise and decay (memory effect)")
    print("- Peak amplitude may differ due to α < 1")
    print("- Fractional order can be estimated from real fMRI data")
