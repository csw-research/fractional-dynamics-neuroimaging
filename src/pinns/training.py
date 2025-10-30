"""
Training utilities for physics-informed neural networks.
"""

import torch
import torch.nn as nn
from torch.optim import Adam, LBFGS
from typing import Optional, Dict, Callable, List
from tqdm import tqdm
import numpy as np


class PINNTrainer:
    """
    Trainer class for physics-informed neural networks.

    Handles:
    - Training loop with progress tracking
    - Loss scheduling and weighting
    - Checkpointing
    - Learning rate scheduling
    - Gradient clipping
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: str = "adam",
        lr: float = 1e-3,
        loss_weights: Optional[Dict[str, float]] = None,
        device: str = "cpu",
    ):
        """
        Initialize trainer.

        Args:
            model: PINN model
            optimizer: "adam" or "lbfgs"
            lr: Learning rate
            loss_weights: Weights for different loss components
            device: Device to train on
        """
        self.model = model.to(device)
        self.device = device

        if optimizer == "adam":
            self.optimizer = Adam(model.parameters(), lr=lr)
        elif optimizer == "lbfgs":
            self.optimizer = LBFGS(
                model.parameters(),
                lr=lr,
                max_iter=20,
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
                history_size=100,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        self.loss_weights = loss_weights or {
            "data": 1.0,
            "pde": 1.0,
            "bc": 1.0,
            "ic": 1.0,
        }

        self.history = {
            "total_loss": [],
            "data_loss": [],
            "pde_loss": [],
            "bc_loss": [],
            "ic_loss": [],
        }

    def train_step(
        self,
        loss_func: Callable,
        *args,
        **kwargs
    ) -> tuple[float, Dict[str, float]]:
        """
        Single training step.

        Args:
            loss_func: Function returning (total_loss, loss_dict)
            args, kwargs: Arguments to loss_func

        Returns:
            (total_loss, loss_dict)
        """
        self.model.train()

        def closure():
            self.optimizer.zero_grad()
            total_loss, loss_dict = loss_func(self.model, *args, **kwargs)
            total_loss.backward()
            return total_loss

        if isinstance(self.optimizer, LBFGS):
            total_loss = self.optimizer.step(closure)
            # Recompute for loss_dict
            with torch.no_grad():
                _, loss_dict = loss_func(self.model, *args, **kwargs)
        else:
            self.optimizer.zero_grad()
            total_loss, loss_dict = loss_func(self.model, *args, **kwargs)
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

        return total_loss.item(), loss_dict

    def train(
        self,
        loss_func: Callable,
        num_epochs: int,
        *args,
        verbose: bool = True,
        checkpoint_freq: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        **kwargs
    ):
        """
        Full training loop.

        Args:
            loss_func: Loss function
            num_epochs: Number of epochs
            verbose: Print progress
            checkpoint_freq: Save checkpoint every N epochs
            checkpoint_path: Path to save checkpoints
        """
        pbar = tqdm(range(num_epochs), disable=not verbose)

        for epoch in pbar:
            total_loss, loss_dict = self.train_step(loss_func, *args, **kwargs)

            # Record history
            self.history["total_loss"].append(total_loss)
            for key, val in loss_dict.items():
                if key in self.history:
                    self.history[f"{key}_loss"].append(val)

            # Update progress bar
            if verbose:
                pbar.set_description(
                    f"Loss: {total_loss:.3e} | "
                    f"Data: {loss_dict.get('data', 0):.3e} | "
                    f"PDE: {loss_dict.get('pde', 0):.3e}"
                )

            # Checkpointing
            if checkpoint_freq and epoch % checkpoint_freq == 0:
                if checkpoint_path:
                    self.save_checkpoint(checkpoint_path, epoch, total_loss)

    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """Save model checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "history": self.history,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
        return checkpoint["epoch"], checkpoint["loss"]


def train_pinn(
    model: nn.Module,
    data_points: Dict[str, torch.Tensor],
    pde_points: Dict[str, torch.Tensor],
    pde_func: Callable,
    alpha: float,
    num_epochs: int = 10000,
    lr: float = 1e-3,
    loss_weights: Optional[Dict[str, float]] = None,
    device: str = "cpu",
    verbose: bool = True,
    bc_points: Optional[Dict[str, torch.Tensor]] = None,
    ic_points: Optional[Dict[str, torch.Tensor]] = None,
) -> PINNTrainer:
    """
    Convenience function for training PINN.

    Args:
        model: PINN model
        data_points: Observation data
        pde_points: Collocation points for PDE
        pde_func: PDE residual function
        alpha: Fractional order
        num_epochs: Number of training epochs
        lr: Learning rate
        loss_weights: Loss component weights
        device: Training device
        verbose: Print progress
        bc_points: Boundary condition points
        ic_points: Initial condition points

    Returns:
        Trained PINNTrainer
    """
    from .losses import combined_pinn_loss

    # Move data to device
    for key in data_points:
        if data_points[key] is not None:
            data_points[key] = data_points[key].to(device)

    for key in pde_points:
        if pde_points[key] is not None:
            pde_points[key] = pde_points[key].to(device)

    if bc_points:
        for key in bc_points:
            if bc_points[key] is not None and isinstance(bc_points[key], torch.Tensor):
                bc_points[key] = bc_points[key].to(device)
    else:
        bc_points = {"x": None}

    if ic_points:
        for key in ic_points:
            if ic_points[key] is not None and isinstance(ic_points[key], torch.Tensor):
                ic_points[key] = ic_points[key].to(device)
    else:
        ic_points = {"x": None}

    # Create trainer
    trainer = PINNTrainer(
        model,
        optimizer="adam",
        lr=lr,
        loss_weights=loss_weights,
        device=device,
    )

    # Define loss function
    def loss_func(model):
        return combined_pinn_loss(
            model,
            data_points,
            pde_points,
            bc_points,
            ic_points,
            pde_func,
            alpha,
            weights=loss_weights,
        )

    # Train
    trainer.train(loss_func, num_epochs, verbose=verbose)

    return trainer


class AdaptiveWeightScheduler:
    """
    Adaptive loss weight scheduler.

    Automatically balances loss components using gradient statistics.

    Reference:
        Wang et al., "Understanding and Mitigating Gradient Flow Pathologies
        in Physics-Informed Neural Networks"
    """

    def __init__(
        self,
        initial_weights: Dict[str, float],
        update_freq: int = 100,
        alpha: float = 0.9,
    ):
        """
        Initialize scheduler.

        Args:
            initial_weights: Initial loss weights
            update_freq: Update weights every N steps
            alpha: EMA smoothing parameter
        """
        self.weights = initial_weights.copy()
        self.update_freq = update_freq
        self.alpha = alpha
        self.step_count = 0

        # Running statistics
        self.grad_norms = {key: [] for key in initial_weights}

    def step(self, model: nn.Module, losses: Dict[str, torch.Tensor]):
        """
        Update weights based on gradient magnitudes.

        Args:
            model: Neural network
            losses: Dictionary of loss components
        """
        self.step_count += 1

        if self.step_count % self.update_freq == 0:
            # Compute gradient norms for each loss component
            for key, loss in losses.items():
                if key in self.weights and loss.requires_grad:
                    # Compute gradient w.r.t. model parameters
                    grads = torch.autograd.grad(
                        loss,
                        model.parameters(),
                        retain_graph=True,
                        allow_unused=True
                    )

                    # Filter out None gradients
                    grads = [g for g in grads if g is not None]

                    if grads:
                        grad_norm = torch.sqrt(sum(torch.sum(g**2) for g in grads))
                        self.grad_norms[key].append(grad_norm.item())

            # Update weights (inverse of gradient norm)
            if all(len(norms) > 0 for norms in self.grad_norms.values()):
                # Average gradient norms
                avg_norms = {key: np.mean(norms[-10:])
                            for key, norms in self.grad_norms.items()
                            if len(norms) > 0}

                # Compute target weights (inverse, with normalization)
                if avg_norms:
                    max_norm = max(avg_norms.values())
                    for key in self.weights:
                        if key in avg_norms and avg_norms[key] > 0:
                            new_weight = max_norm / avg_norms[key]
                            # EMA update
                            self.weights[key] = (
                                self.alpha * self.weights[key] +
                                (1 - self.alpha) * new_weight
                            )

    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return self.weights.copy()


class CurriculumLearning:
    """
    Curriculum learning for PINNs.

    Gradually increases problem difficulty:
    1. Start with coarse collocation points
    2. Add more points over time
    3. Increase PDE loss weight
    """

    def __init__(
        self,
        initial_pde_points: int,
        max_pde_points: int,
        growth_rate: float = 1.1,
        pde_weight_schedule: str = "linear",
    ):
        """
        Initialize curriculum.

        Args:
            initial_pde_points: Starting number of collocation points
            max_pde_points: Maximum number of collocation points
            growth_rate: Multiplicative growth rate
            pde_weight_schedule: "linear", "exponential", or "constant"
        """
        self.initial_pde_points = initial_pde_points
        self.max_pde_points = max_pde_points
        self.growth_rate = growth_rate
        self.pde_weight_schedule = pde_weight_schedule

        self.current_pde_points = initial_pde_points
        self.epoch = 0

    def step(self, epoch: int) -> tuple[int, float]:
        """
        Update curriculum.

        Args:
            epoch: Current epoch

        Returns:
            (num_pde_points, pde_weight)
        """
        self.epoch = epoch

        # Update number of points
        if self.current_pde_points < self.max_pde_points:
            self.current_pde_points = min(
                int(self.current_pde_points * self.growth_rate),
                self.max_pde_points
            )

        # Update PDE weight
        progress = self.current_pde_points / self.max_pde_points

        if self.pde_weight_schedule == "linear":
            pde_weight = progress
        elif self.pde_weight_schedule == "exponential":
            pde_weight = progress ** 2
        else:  # constant
            pde_weight = 1.0

        return self.current_pde_points, pde_weight


def early_stopping(
    loss_history: List[float],
    patience: int = 100,
    min_delta: float = 1e-6
) -> bool:
    """
    Check if training should stop early.

    Args:
        loss_history: List of losses
        patience: Number of epochs to wait
        min_delta: Minimum improvement threshold

    Returns:
        True if should stop
    """
    if len(loss_history) < patience + 1:
        return False

    recent_losses = loss_history[-patience:]
    best_recent = min(recent_losses)
    current = loss_history[-1]

    # Check if no improvement
    improvement = best_recent - current

    return improvement < min_delta
