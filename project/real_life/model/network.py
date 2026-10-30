"""
project/real_life/model/network.py
Physics-Informed Neural Network (PINN) architecture for the Real-Life track.
Features:
- Separate feedstock and process branch encoders fused before output heads.
- Heteroscedastic Gaussian output head (predicts mean and log-variance).
- Differentiable parallel kinetic ODE layer.
- Proximate-to-biopolymer learnable mapper if proximate mode is used.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.ode_integrator import KineticDecompositionODE


class RealLifePINN(nn.Module):
    """
    PINN architecture for real-world biochar production.
    """

    def __init__(
        self,
        feedstock_dim: int,
        process_dim: int = 3,
        feedstock_hidden: Tuple[int, ...] = (64, 32),
        process_hidden: Tuple[int, ...] = (32, 16),
        fusion_hidden: Tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
        mode: str = "biopolymers", # 'biopolymers' or 'proximate'
    ):
        super().__init__()
        self.mode = mode

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
        self.feedstock_latent_dim = feedstock_hidden[-1]

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
        self.process_latent_dim = process_hidden[-1]

        # Fusion branch
        fuse_dim = self.feedstock_latent_dim + self.process_latent_dim
        fuse_layers = []
        for h in fusion_hidden:
            fuse_layers.append(nn.Linear(fuse_dim, h))
            fuse_layers.append(nn.LayerNorm(h))
            fuse_layers.append(nn.SiLU())
            fuse_layers.append(nn.Dropout(dropout))
            fuse_dim = h
        self.fusion_network = nn.Sequential(*fuse_layers)

        # Heteroscedastic output head: mean and log-variance
        self.mean_head = nn.Linear(fusion_hidden[-1], 1)
        self.log_var_head = nn.Linear(fusion_hidden[-1], 1)

        # Initialize output heads near empirical prior (yield ~ 35%, std ~ 5%)
        nn.init.constant_(self.mean_head.bias, 35.0)
        nn.init.normal_(self.mean_head.weight, std=0.01)
        nn.init.constant_(self.log_var_head.bias, 3.2)
        nn.init.normal_(self.log_var_head.weight, std=0.01)

        # Kinetics ODE layer (stateless parallel Arrhenius model)
        self.kinetics_layer = KineticDecompositionODE(learnable=False)

        # Proximate to biopolymer mapper if operating in proximate fallback mode
        if self.mode == "proximate":
            self.proximate_mapper = nn.Sequential(
                nn.Linear(feedstock_dim, 16),
                nn.SiLU(),
                nn.Linear(16, 4),
                nn.Softmax(dim=-1),
            )
        else:
            self.proximate_mapper = None

    def forward(
        self,
        feedstock_x: torch.Tensor,
        process_x: torch.Tensor,
        raw_process: torch.Tensor,
        raw_biopolymers: Optional[torch.Tensor] = None,
        ash_weight: Optional[torch.Tensor] = None,
        compute_kinetics: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            feedstock_x: Transformed feedstock composition tensor.
            process_x: Normalized process conditions (T, RT, HR).
            raw_process: Unscaled process conditions [T (°C), RT (min), HR (°C/min)].
            raw_biopolymers: Raw biopolymer fractions (if biopolymers mode).
            ash_weight: Ash fraction.
            compute_kinetics: Whether to evaluate ODE kinetics (set False during collocation).
        Returns:
            Dictionary containing predicted yield, variance, std, and optional kinetic yield.
        """
        # Encode branches
        z_feedstock = self.feedstock_encoder(feedstock_x)
        z_process = self.process_encoder(process_x)

        # Latent fusion
        z_fused = torch.cat([z_feedstock, z_process], dim=-1)
        features = self.fusion_network(z_fused)

        # Heteroscedastic outputs
        yield_mean = self.mean_head(features)
        yield_log_var = self.log_var_head(features)
        # Numerical clamping for stability
        yield_log_var = torch.clamp(yield_log_var, min=-10.0, max=5.0)
        yield_std = torch.sqrt(torch.exp(yield_log_var))

        kinetic_yield = None
        conversions = None

        if compute_kinetics:
            # Kinetic ODE integration
            if self.mode == "proximate" and self.proximate_mapper is not None:
                bio_fractions = self.proximate_mapper(feedstock_x)
            else:
                bio_fractions = raw_biopolymers if raw_biopolymers is not None else torch.ones((feedstock_x.shape[0], 4), device=feedstock_x.device) * 0.25

            T_peak_C = raw_process[:, 0:1]
            RT_min = raw_process[:, 1:2]
            HR_C_per_min = raw_process[:, 2:3]

            kinetic_yield, conversions = self.kinetics_layer(
                biopolymer_weights=bio_fractions,
                T_peak_C=T_peak_C,
                residence_time_min=RT_min,
                heating_rate_C_per_min=HR_C_per_min,
                ash_weight=ash_weight,
            )

        return {
            "yield_mean": yield_mean,
            "yield_log_var": yield_log_var,
            "yield_std": yield_std,
            "kinetic_yield": kinetic_yield,
            "conversions": conversions,
        }
