"""
project/research/model/network.py
Physics-Informed Neural Network for the Research track.
Supports:
- Multi-block feedstock fusion (Biopolymer + Proximate + Ultimate ILR representations).
- Multi-target heteroscedastic heads (Yield, N-char, SSA-char).
- Learnable kinetic decomposition ODE layer for calibration.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.ode_integrator import KineticDecompositionODE


class ResearchPINN(nn.Module):
    """Multi-target PINN architecture with calibrated ODE kinetics."""

    def __init__(
        self,
        feedstock_dim: int,
        process_dim: int = 3,
        num_targets: int = 3, # [Yield, N-char, SSA-char]
        feedstock_hidden: Tuple[int, ...] = (128, 64),
        process_hidden: Tuple[int, ...] = (32, 16),
        fusion_hidden: Tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
        learnable_kinetics: bool = True,
    ):
        super().__init__()
        self.num_targets = num_targets

        # Feedstock branch
        f_layers = []
        in_dim = feedstock_dim
        for h in feedstock_hidden:
            f_layers.append(nn.Linear(in_dim, h))
            f_layers.append(nn.LayerNorm(h))
            f_layers.append(nn.SiLU())
            f_layers.append(nn.Dropout(dropout))
            in_dim = h
        self.feedstock_encoder = nn.Sequential(*f_layers)

        # Process branch
        p_layers = []
        in_dim = process_dim
        for h in process_hidden:
            p_layers.append(nn.Linear(in_dim, h))
            p_layers.append(nn.LayerNorm(h))
            p_layers.append(nn.SiLU())
            p_layers.append(nn.Dropout(dropout))
            in_dim = h
        self.process_encoder = nn.Sequential(*p_layers)

        # Fusion network
        fuse_dim = feedstock_hidden[-1] + process_hidden[-1]
        fuse_layers = []
        for h in fusion_hidden:
            fuse_layers.append(nn.Linear(fuse_dim, h))
            fuse_layers.append(nn.LayerNorm(h))
            fuse_layers.append(nn.SiLU())
            fuse_layers.append(nn.Dropout(dropout))
            fuse_dim = h
        self.fusion_network = nn.Sequential(*fuse_layers)

        # Multi-target heads (mean and log_var for each target)
        self.mean_head = nn.Linear(fusion_hidden[-1], num_targets)
        self.log_var_head = nn.Linear(fusion_hidden[-1], num_targets)

        # Prior initializations for targets: [Yield ~ 35 wt%, N-char ~ 1.5 wt%, SSA ~ 50 m2/g]
        with torch.no_grad():
            self.mean_head.bias.copy_(torch.tensor([35.0, 1.5, 50.0][:num_targets]))
            nn.init.normal_(self.mean_head.weight, std=0.01)
            self.log_var_head.bias.copy_(torch.tensor([3.0, 0.5, 5.0][:num_targets]))
            nn.init.normal_(self.log_var_head.weight, std=0.01)

        # Kinetic ODE layer (learnable kinetic priors in research track)
        self.kinetics_layer = KineticDecompositionODE(learnable=learnable_kinetics)

    def forward(
        self,
        feedstock_x: torch.Tensor,
        process_x: torch.Tensor,
        raw_process: torch.Tensor,
        raw_biopolymers: torch.Tensor,
        ash_weight: Optional[torch.Tensor] = None,
        compute_kinetics: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Returns:
            - 'means': Tensor of shape (Batch, num_targets)
            - 'log_vars': Tensor of shape (Batch, num_targets)
            - 'stds': Tensor of shape (Batch, num_targets)
            - 'kinetic_yield': Analytical ODE yield (Batch, 1) or None
        """
        z_f = self.feedstock_encoder(feedstock_x)
        z_p = self.process_encoder(process_x)

        z_fused = torch.cat([z_f, z_p], dim=-1)
        latent = self.fusion_network(z_fused)

        means = self.mean_head(latent)
        log_vars = torch.clamp(self.log_var_head(latent), min=-10.0, max=5.0)
        stds = torch.sqrt(torch.exp(log_vars))

        kinetic_yield = None
        conversions = None

        if compute_kinetics:
            T_peak_C = raw_process[:, 0:1]
            RT_min = raw_process[:, 1:2]
            HR_C_per_min = raw_process[:, 2:3]

            kinetic_yield, conversions = self.kinetics_layer(
                biopolymer_weights=raw_biopolymers,
                T_peak_C=T_peak_C,
                residence_time_min=RT_min,
                heating_rate_C_per_min=HR_C_per_min,
                ash_weight=ash_weight,
            )

        return {
            "means": means,
            "log_vars": log_vars,
            "stds": stds,
            "kinetic_yield": kinetic_yield,
            "conversions": conversions,
        }
