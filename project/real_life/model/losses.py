"""
project/real_life/model/losses.py
Physics-informed loss functions and adaptive multi-task weighting.
Includes:
- Heteroscedastic Gaussian NLL loss
- Kinetic ODE consistency loss
- Monotonicity constraint loss on collocation points (dY/dT <= 0)
- Kendall & Gal homoscedastic uncertainty adaptive weighting
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn


class HeteroscedasticLoss(nn.Module):
    """Gaussian Negative Log-Likelihood with predicted log-variance s = ln(sigma^2)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, log_var: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        inv_var = torch.exp(-log_var)
        squared_err = (y_true - y_pred) ** 2
        loss = 0.5 * (inv_var * squared_err + log_var)
        return torch.mean(loss)


class PhysicsInformedLoss(nn.Module):
    """
    Combined loss integrating:
    1. Supervised heteroscedastic NLL
    2. Kinetic consistency |y_nn - y_kinetic|^2
    3. Monotonicity violation ReLU(d(y_nn)/dT)^2 on collocation points
    """

    def __init__(
        self,
        adaptive: bool = True,
        initial_lambda_kin: float = 0.4,
        initial_lambda_mono: float = 0.2,
    ):
        super().__init__()
        self.nll_criterion = HeteroscedasticLoss()
        self.adaptive = adaptive
        self.lambda_mono = initial_lambda_mono

        if adaptive:
            # Kendall & Gal homoscedastic task log-variances for [data_nll, kinetic_loss]
            self.task_log_vars = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        else:
            self.lambda_data = 1.0
            self.lambda_kin = initial_lambda_kin

    def forward(
        self,
        pred_dict: Dict[str, torch.Tensor],
        target_yield: torch.Tensor,
        collocation_grad_T: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the total loss and individual loss components.
        """
        y_mean = pred_dict["yield_mean"]
        y_log_var = pred_dict["yield_log_var"]
        y_kinetic = pred_dict["kinetic_yield"]

        # 1. Supervised NLL
        loss_data = self.nll_criterion(y_mean, y_log_var, target_yield)

        # 2. Kinetic Consistency Loss
        loss_kinetics = torch.mean((y_mean - y_kinetic.detach()) ** 2)

        # 3. Monotonicity Loss (temperature gradient should be non-positive: dY/dT <= 0)
        if collocation_grad_T is not None and collocation_grad_T.numel() > 0:
            mono_violations = torch.relu(collocation_grad_T)
            loss_mono = torch.mean(mono_violations ** 2)
        else:
            loss_mono = torch.tensor(0.0, device=target_yield.device)

        if self.adaptive:
            # Clamped Kendall-Gal adaptive loss formulation
            s_data = torch.clamp(self.task_log_vars[0], -4.0, 4.0)
            s_kin = torch.clamp(self.task_log_vars[1], -4.0, 4.0)
            
            total_loss = (
                torch.exp(-s_data) * loss_data + 0.5 * s_data
                + torch.exp(-s_kin) * loss_kinetics + 0.5 * s_kin
                + self.lambda_mono * loss_mono
            )
        else:
            total_loss = (
                self.lambda_data * loss_data
                + self.lambda_kin * loss_kinetics
                + self.lambda_mono * loss_mono
            )

        loss_breakdown = {
            "total_loss": float(total_loss.item()),
            "loss_data": float(loss_data.item()),
            "loss_kinetics": float(loss_kinetics.item()),
            "loss_mono": float(loss_mono.item()),
        }

        return total_loss, loss_breakdown
