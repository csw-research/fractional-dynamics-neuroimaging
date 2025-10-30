"""
Bayesian PINNs for uncertainty quantification.

Implements:
1. Variational inference for parameter uncertainty
2. Monte Carlo dropout
3. Ensemble methods
"""

import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with weight uncertainty.

    Uses local reparameterization trick for efficiency.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        # Weight parameters (mean and log-variance)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 5)

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.zeros(out_features) - 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with weight sampling.
        """
        if self.training:
            # Sample weights
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight = self.weight_mu + weight_std * torch.randn_like(weight_std)

            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
        else:
            # Use mean weights for inference
            weight = self.weight_mu
            bias = self.bias_mu

        return nn.functional.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """
        KL divergence between posterior and prior.

        Assumes prior: N(0, 1)
        """
        kl_weight = 0.5 * torch.sum(
            torch.exp(self.weight_logvar) + self.weight_mu**2 - 1 - self.weight_logvar
        )

        kl_bias = 0.5 * torch.sum(
            torch.exp(self.bias_logvar) + self.bias_mu**2 - 1 - self.bias_logvar
        )

        return kl_weight + kl_bias


class BayesianPINN(nn.Module):
    """
    Bayesian PINN with uncertainty quantification.

    Uses variational inference to learn posterior distribution over weights.
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        hidden_layers: List[int] = [64, 64, 64],
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(BayesianLinear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim

        layers.append(BayesianLinear(prev_dim, output_dim))

        self.network = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Bayesian layers."""
        for layer in self.network:
            x = layer(x)
        return x

    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence across all Bayesian layers."""
        kl = 0.0
        for layer in self.network:
            if isinstance(layer, BayesianLinear):
                kl += layer.kl_divergence()
        return kl

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prediction with uncertainty estimates.

        Args:
            x: Input points
            num_samples: Number of Monte Carlo samples

        Returns:
            (mean, std): Mean prediction and standard deviation
        """
        self.train()  # Enable dropout/sampling

        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)

        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)

        self.eval()

        return mean, std


class MCDropoutPINN(nn.Module):
    """
    PINN with Monte Carlo dropout for uncertainty quantification.

    Simpler alternative to full Bayesian inference.
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        hidden_layers: List[int] = [64, 64, 64],
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MC dropout prediction.

        Args:
            x: Input points
            num_samples: Number of forward passes

        Returns:
            (mean, std)
        """
        self.train()  # Keep dropout active

        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)

        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)

        return mean, std


class EnsemblePINN:
    """
    Ensemble of PINNs for uncertainty quantification.

    Trains multiple models with different initializations.
    """

    def __init__(
        self,
        model_class,
        num_models: int = 5,
        **model_kwargs
    ):
        """
        Initialize ensemble.

        Args:
            model_class: PINN class to instantiate
            num_models: Number of ensemble members
            model_kwargs: Arguments for model_class
        """
        self.models = [model_class(**model_kwargs) for _ in range(num_models)]
        self.num_models = num_models

    def train_ensemble(self, train_func, **train_kwargs):
        """
        Train all ensemble members.

        Args:
            train_func: Function that trains a single model
            train_kwargs: Arguments for train_func
        """
        for i, model in enumerate(self.models):
            print(f"Training ensemble member {i+1}/{self.num_models}...")
            train_func(model, **train_kwargs)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ensemble prediction.

        Args:
            x: Input points

        Returns:
            (mean, std)
        """
        predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)

        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)

        return mean, std


def uncertainty_quantification(
    model: nn.Module,
    x: torch.Tensor,
    method: str = "mc_dropout",
    num_samples: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generic uncertainty quantification interface.

    Args:
        model: Trained PINN
        x: Input points
        method: "mc_dropout", "bayesian", or "ensemble"
        num_samples: Number of samples for MC methods

    Returns:
        (mean, std)
    """
    if hasattr(model, "predict_with_uncertainty"):
        return model.predict_with_uncertainty(x, num_samples)
    else:
        # Fallback: simple prediction without uncertainty
        model.eval()
        with torch.no_grad():
            pred = model(x)
        return pred, torch.zeros_like(pred)


def confidence_interval(
    mean: torch.Tensor,
    std: torch.Tensor,
    confidence: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute confidence intervals.

    Args:
        mean: Predicted mean
        std: Predicted standard deviation
        confidence: Confidence level (e.g., 0.95 for 95%)

    Returns:
        (lower_bound, upper_bound)
    """
    from scipy.stats import norm

    z_score = norm.ppf((1 + confidence) / 2)

    lower = mean - z_score * std
    upper = mean + z_score * std

    return lower, upper
