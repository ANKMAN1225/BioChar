"""
project/research/model/losses.py
Masked heteroscedastic multi-target loss and physics regularizers for Research track.
Handles partially unobserved targets via binary masks without zero-filling or synthetic imputation.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn


class MaskedHeteroscedasticLoss(nn.Module):
    """
    Gaussian NLL computed strictly over observed targets:
    L = (1 / sum(mask)) * sum_ij [ mask_ij * 0.5 * (exp(-s_ij)*(y_ij - mu_ij)^2 + s_ij) ]
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        means: torch.Tensor,
        log_vars: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        inv_var = torch.exp(-log_vars)
        sq_err = (targets - means) ** 2
        element_nll = 0.5 * (inv_var * sq_err + log_vars)
        masked_nll = element_nll * mask
        num_observed = torch.clamp(torch.sum(mask), min=1.0)
        return torch.sum(masked_nll) / num_observed


class ResearchPhysicsLoss(nn.Module):
    """
    Multi-task loss for research track combining:
    1. Masked heteroscedastic loss for all measured properties
    2. Kinetic consistency on solid biochar yield
    3. Monotonicity violation penalty on collocation points (dYield/dT <= 0)
    4. Kendall & Gal adaptive uncertainty task weights
    """

    def __init__(self, adaptive: bool = True, lambda_mono: float = 0.2):
        super().__init__()
        self.masked_nll = MaskedHeteroscedasticLoss()
        self.adaptive = adaptive
        self.lambda_mono = lambda_mono

        if adaptive:
            # log-variances for [data_nll, kinetic_loss]
            self.task_log_vars = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        else:
            self.task_log_vars = None

    def forward(
        self,
        pred_dict: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        target_mask: torch.Tensor,
        collocation_grad_T: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        means = pred_dict["means"]
        log_vars = pred_dict["log_vars"]
        kinetic_yield = pred_dict["kinetic_yield"]

        # 1. Masked Data Loss across all properties
        loss_data = self.masked_nll(means, log_vars, targets, target_mask)

        # 2. Kinetic Consistency on Yield (index 0)
        yield_pred = means[:, 0:1]
        loss_kinetics = torch.mean((yield_pred - kinetic_yield) ** 2)

        # 3. Monotonicity violation on collocation points
        if collocation_grad_T is not None and collocation_grad_T.numel() > 0:
            mono_violations = torch.relu(collocation_grad_T)
            loss_mono = torch.mean(mono_violations ** 2)
        else:
            loss_mono = torch.tensor(0.0, device=targets.device)

        if self.adaptive:
            s_data = torch.clamp(self.task_log_vars[0], -4.0, 4.0)
            s_kin = torch.clamp(self.task_log_vars[1], -4.0, 4.0)
            total_loss = (
                torch.exp(-s_data) * loss_data + 0.5 * s_data
                + torch.exp(-s_kin) * loss_kinetics + 0.5 * s_kin
                + self.lambda_mono * loss_mono
            )
        else:
            total_loss = loss_data + 0.4 * loss_kinetics + self.lambda_mono * loss_mono

        breakdown = {
            "total_loss": float(total_loss.item()),
            "loss_data": float(loss_data.item()),
            "loss_kinetics": float(loss_kinetics.item()),
            "loss_mono": float(loss_mono.item()),
        }

        return total_loss, breakdown
