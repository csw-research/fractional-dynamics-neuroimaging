"""
Physics-informed neural network architectures for fractional dynamics.

Key architectures:
1. FractionalPINN - Base PINN with fractional derivative support
2. DeepONet - Operator learning for fractional PDEs
3. MultiFidelityPINN - Transfer learning across resolutions/subjects
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, List, Tuple
import numpy as np


class FractionalPINN(nn.Module):
    """
    Physics-informed neural network for fractional differential equations.

    Architecture:
        Input (x, t) → [Dense layers with activation] → Output u(x, t)

    The network is trained to satisfy:
        1. Data: u(x, t) ≈ u_observed at measurement points
        2. PDE: D^α u - f(x, t, u) ≈ 0 at collocation points
        3. BC: u satisfies boundary conditions
        4. IC: u(x, 0) = u₀(x)

    Args:
        input_dim: Input dimensionality (e.g., 2 for (x,t))
        output_dim: Output dimensionality (typically 1 for scalar fields)
        hidden_layers: List of hidden layer sizes
        activation: Activation function
        fractional_order: α for fractional derivatives (can be learnable)
        learn_fractional_order: Whether to learn α as a parameter
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        hidden_layers: List[int] = [64, 64, 64, 64],
        activation: str = "tanh",
        fractional_order: float = 0.5,
        learn_fractional_order: bool = False,
        use_fourier_features: bool = False,
        fourier_scale: float = 1.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_fourier_features = use_fourier_features

        # Fourier feature mapping (helps with high-frequency functions)
        if use_fourier_features:
            self.fourier_dim = 256
            self.B = torch.randn(input_dim, self.fourier_dim // 2) * fourier_scale
            input_dim_net = self.fourier_dim
        else:
            input_dim_net = input_dim

        # Build network
        layers = []
        prev_dim = input_dim_net

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Fractional order parameter
        if learn_fractional_order:
            # Use sigmoid to constrain to (0, 1)
            self.alpha_raw = nn.Parameter(torch.tensor(fractional_order))
        else:
            self.register_buffer("alpha_raw", torch.tensor(fractional_order))

        self.learn_fractional_order = learn_fractional_order

        # Initialize weights
        self._initialize_weights()

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "sin": lambda x: torch.sin(x),
        }

        if name in activations:
            return activations[name] if isinstance(activations[name], nn.Module) \
                   else nn.Module()  # For lambda functions
        else:
            raise ValueError(f"Unknown activation: {name}")

    def _initialize_weights(self):
        """Xavier initialization for better training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def fourier_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier feature mapping: [cos(2π Bx), sin(2π Bx)]

        Helps network learn high-frequency functions.
        Reference: Tancik et al., "Fourier Features Let Networks Learn..."
        """
        x_proj = 2 * np.pi * x @ self.B.to(x.device)
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, input_dim)

        Returns:
            Output tensor, shape (batch, output_dim)
        """
        if self.use_fourier_features:
            x = self.fourier_features(x)

        return self.network(x)

    @property
    def alpha(self) -> torch.Tensor:
        """Get fractional order (constrained to (0, 1))."""
        if self.learn_fractional_order:
            return torch.sigmoid(self.alpha_raw)
        else:
            return self.alpha_raw

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method for prediction."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""

    def __init__(self, dim: int, activation: nn.Module = nn.Tanh()):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        out = out + residual  # Skip connection
        return self.activation(out)


class DeepResidualPINN(FractionalPINN):
    """
    Deep residual PINN for complex problems.

    Uses residual connections to enable training of very deep networks.
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        hidden_dim: int = 64,
        num_blocks: int = 8,
        fractional_order: float = 0.5,
        **kwargs
    ):
        # Don't call parent __init__, build custom architecture
        nn.Module.__init__(self)

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Fractional order
        self.register_buffer("alpha_raw", torch.tensor(fractional_order))
        self.learn_fractional_order = False

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.input_layer(x))

        for block in self.blocks:
            x = block(x)

        return self.output_layer(x)


class DeepONet(nn.Module):
    """
    Deep Operator Network for learning operators.

    Learns mappings between function spaces: G: u₀ → u(t)
    Useful for parametric PDEs and transfer learning.

    Architecture:
        Branch net: u₀(x) → b(u₀)
        Trunk net: (x, t) → t(x, t)
        Output: G(u₀)(x, t) = b · t (dot product)

    Reference:
        Lu et al., "Learning nonlinear operators via DeepONet"
    """

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        basis_dim: int = 100,
        branch_hidden: List[int] = [64, 64],
        trunk_hidden: List[int] = [64, 64],
    ):
        super().__init__()

        # Branch network (encodes input function)
        branch_layers = []
        prev_dim = branch_input_dim
        for hidden_dim in branch_hidden:
            branch_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh(),
            ])
            prev_dim = hidden_dim
        branch_layers.append(nn.Linear(prev_dim, basis_dim))
        self.branch_net = nn.Sequential(*branch_layers)

        # Trunk network (encodes query points)
        trunk_layers = []
        prev_dim = trunk_input_dim
        for hidden_dim in trunk_hidden:
            trunk_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh(),
            ])
            prev_dim = hidden_dim
        trunk_layers.append(nn.Linear(prev_dim, basis_dim))
        self.trunk_net = nn.Sequential(*trunk_layers)

        # Bias term
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        branch_input: torch.Tensor,
        trunk_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            branch_input: Input function values, shape (batch, branch_input_dim)
            trunk_input: Query points, shape (batch, trunk_input_dim)

        Returns:
            Output values, shape (batch, 1)
        """
        b = self.branch_net(branch_input)  # (batch, basis_dim)
        t = self.trunk_net(trunk_input)    # (batch, basis_dim)

        # Dot product
        output = torch.sum(b * t, dim=1, keepdim=True) + self.bias

        return output


class MultiFidelityPINN(nn.Module):
    """
    Multi-fidelity PINN for transfer learning.

    Combines low-fidelity (cheap simulations) and high-fidelity (expensive
    measurements) data.

    Architecture:
        u_high(x) = u_low(x) + Δu(x)

    where u_low is pretrained on abundant low-fidelity data, and Δu is
    learned to correct for high-fidelity data.
    """

    def __init__(
        self,
        low_fidelity_model: FractionalPINN,
        correction_hidden: List[int] = [32, 32],
    ):
        super().__init__()

        self.low_fidelity_model = low_fidelity_model

        # Freeze low-fidelity model
        for param in self.low_fidelity_model.parameters():
            param.requires_grad = False

        # Correction network
        input_dim = low_fidelity_model.input_dim
        output_dim = low_fidelity_model.output_dim

        correction_layers = []
        prev_dim = input_dim
        for hidden_dim in correction_hidden:
            correction_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh(),
            ])
            prev_dim = hidden_dim
        correction_layers.append(nn.Linear(prev_dim, output_dim))

        self.correction_net = nn.Sequential(*correction_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: u_high = u_low + Δu
        """
        u_low = self.low_fidelity_model(x)
        delta_u = self.correction_net(x)

        return u_low + delta_u

    def unfreeze_low_fidelity(self):
        """Unfreeze low-fidelity model for fine-tuning."""
        for param in self.low_fidelity_model.parameters():
            param.requires_grad = True


class AdaptivePINN(FractionalPINN):
    """
    Adaptive PINN with automatic collocation point refinement.

    Adds more collocation points in regions where PDE residual is large.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collocation_points = None
        self.residuals = None

    def update_collocation_points(
        self,
        domain_bounds: Tuple[Tuple[float, float], ...],
        num_points: int,
        top_k_percent: float = 0.2,
    ):
        """
        Adaptively refine collocation points based on residual magnitudes.

        Args:
            domain_bounds: ((x_min, x_max), (t_min, t_max), ...)
            num_points: Total number of collocation points
            top_k_percent: Fraction of points to add near high residuals
        """
        if self.residuals is None:
            # Initial uniform sampling
            from src.fractional.utils import sample_random_points
            self.collocation_points = sample_random_points(
                domain_bounds,
                num_points,
                dim=len(domain_bounds),
                method="latin_hypercube"
            )
        else:
            # Adaptive refinement
            num_refine = int(num_points * top_k_percent)
            num_uniform = num_points - num_refine

            # Sample uniformly
            from src.fractional.utils import sample_random_points
            uniform_points = sample_random_points(
                domain_bounds,
                num_uniform,
                dim=len(domain_bounds),
                method="uniform"
            )

            # Sample near high residuals (importance sampling)
            if self.collocation_points is not None:
                residual_magnitudes = torch.abs(self.residuals)
                probs = residual_magnitudes / torch.sum(residual_magnitudes)

                indices = torch.multinomial(probs.flatten(), num_refine, replacement=True)
                refine_points = self.collocation_points[indices]

                # Add noise
                noise = torch.randn_like(refine_points) * 0.1
                refine_points = refine_points + noise

                # Combine
                self.collocation_points = torch.cat([uniform_points, refine_points], dim=0)
            else:
                self.collocation_points = uniform_points

        return self.collocation_points
