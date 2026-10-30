"""
project/utils/ode_integrator.py
Differentiable parallel multi-component kinetic ODE solver for biomass pyrolysis.
Implements the distributed activation energy / parallel reaction model for:
- Hemicellulose
- Cellulose
- Lignin
- Extractives
All kinetic default parameters are marked as: [PLACEHOLDER - VERIFY AGAINST LITERATURE].
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

# Universal Gas Constant R in kJ / (mol * K)
R_GAS_CONST = 8.314462618e-3


# Standard literature-informed initial parameters (placeholder - verify against literature)
DEFAULT_KINETIC_PARAMS = {
    "hemicellulose": {
        "log_A": 25.328,   # ln(A) where A ~ 1.0e11 s^-1 [PLACEHOLDER - VERIFY]
        "Ea": 105.0,       # kJ/mol [PLACEHOLDER - VERIFY]
        "n": 1.0,          # reaction order [PLACEHOLDER - VERIFY]
        "char_inf": 0.20,  # asymptotic char fraction [PLACEHOLDER - VERIFY]
    },
    "cellulose": {
        "log_A": 32.236,   # ln(A) where A ~ 1.0e14 s^-1 [PLACEHOLDER - VERIFY]
        "Ea": 195.0,       # kJ/mol [PLACEHOLDER - VERIFY]
        "n": 1.0,          # reaction order [PLACEHOLDER - VERIFY]
        "char_inf": 0.12,  # asymptotic char fraction [PLACEHOLDER - VERIFY]
    },
    "lignin": {
        "log_A": 13.815,   # ln(A) where A ~ 1.0e6 s^-1 [PLACEHOLDER - VERIFY]
        "Ea": 65.0,        # kJ/mol [PLACEHOLDER - VERIFY]
        "n": 2.5,          # reaction order [PLACEHOLDER - VERIFY]
        "char_inf": 0.46,  # asymptotic char fraction [PLACEHOLDER - VERIFY]
    },
    "extractives": {
        "log_A": 20.030,   # ln(A) where A ~ 5.0e8 s^-1 [PLACEHOLDER - VERIFY]
        "Ea": 80.0,        # kJ/mol [PLACEHOLDER - VERIFY]
        "n": 1.2,          # reaction order [PLACEHOLDER - VERIFY]
        "char_inf": 0.10,  # asymptotic char fraction [PLACEHOLDER - VERIFY]
    },
}


class KineticDecompositionODE(nn.Module):
    """
    Differentiable ODE layer modeling multi-component thermal devolatilization.
    Can operate in frozen (stateless) mode or learnable calibrated mode.
    """

    def __init__(
        self,
        components_config: Optional[Dict[str, Dict[str, float]]] = None,
        learnable: bool = False,
        num_steps: int = 30,
    ):
        super().__init__()
        self.learnable = learnable
        self.num_steps = num_steps
        cfg = components_config or DEFAULT_KINETIC_PARAMS

        comp_keys = ["hemicellulose", "cellulose", "lignin", "extractives"]
        self.comp_keys = comp_keys

        log_A_list = [cfg[k].get("log_A", 20.0) for k in comp_keys]
        Ea_list = [cfg[k].get("Ea", 100.0) for k in comp_keys]
        n_list = [cfg[k].get("n", 1.0) for k in comp_keys]
        char_inf_list = [cfg[k].get("char_inf", 0.2) for k in comp_keys]

        if learnable:
            self.log_A = nn.Parameter(torch.tensor(log_A_list, dtype=torch.float32))
            self.Ea = nn.Parameter(torch.tensor(Ea_list, dtype=torch.float32))
            self.n_order = nn.Parameter(torch.tensor(n_list, dtype=torch.float32))
            self.char_inf = nn.Parameter(torch.tensor(char_inf_list, dtype=torch.float32))
        else:
            self.register_buffer("log_A", torch.tensor(log_A_list, dtype=torch.float32))
            self.register_buffer("Ea", torch.tensor(Ea_list, dtype=torch.float32))
            self.register_buffer("n_order", torch.tensor(n_list, dtype=torch.float32))
            self.register_buffer("char_inf", torch.tensor(char_inf_list, dtype=torch.float32))

    def get_rate_constant(self, T_K: torch.Tensor) -> torch.Tensor:
        """
        Calculates Arrhenius rate constants k_i(T) = exp(log_A_i - Ea_i / (R * T)).
        T_K shape: (Batch, 1) or scalar.
        Returns shape: (Batch, num_components).
        """
        # Clamp temperature to avoid numerical division by zero
        T_safe = torch.clamp(T_K, min=250.0)
        # log_A shape: (num_components), Ea shape: (num_components)
        # exponent: log_A - Ea / (R * T)
        exponent = self.log_A.unsqueeze(0) - (self.Ea.unsqueeze(0) / (R_GAS_CONST * T_safe))
        # Clamp exponent to prevent overflow
        exponent_clipped = torch.clamp(exponent, min=-50.0, max=50.0)
        return torch.exp(exponent_clipped)

    def forward(
        self,
        biopolymer_weights: torch.Tensor,
        T_peak_C: torch.Tensor,
        residence_time_min: torch.Tensor,
        heating_rate_C_per_min: torch.Tensor,
        ash_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrates reaction kinetics over the explicit thermal history.

        Args:
            biopolymer_weights: Tensor of shape (Batch, 4) for [Hemi, Cel, Lig, Ext], normalized on organic basis.
            T_peak_C: Tensor of shape (Batch, 1) in °C.
            residence_time_min: Tensor of shape (Batch, 1) in minutes.
            heating_rate_C_per_min: Tensor of shape (Batch, 1) in °C/min.
            ash_weight: Optional Tensor of shape (Batch, 1) representing inert ash fraction [0, 1].

        Returns:
            predicted_char_yield_pct: Tensor of shape (Batch, 1) in wt% [0, 100].
            final_conversions: Tensor of shape (Batch, 4) devolatilization conversions alpha_i in [0, 1].
        """
        batch_size = biopolymer_weights.shape[0]
        device = biopolymer_weights.device

        T0_K = 298.15
        T_peak_K = T_peak_C + 273.15
        hr_K_per_s = torch.clamp(heating_rate_C_per_min / 60.0, min=1e-4)
        rt_s = torch.clamp(residence_time_min * 60.0, min=0.0)

        ramp_s = torch.clamp((T_peak_K - T0_K) / hr_K_per_s, min=1.0)
        total_time_s = ramp_s + rt_s

        # Adaptive time grid integration via Runge-Kutta 4th order (RK4)
        dt = total_time_s / float(self.num_steps)  # shape (Batch, 1)

        # alpha state: (Batch, 4) initialized to 0.0
        alpha = torch.zeros(batch_size, 4, device=device, dtype=biopolymer_weights.dtype)

        # Disables graph construction when kinetics are frozen/stateless
        with torch.set_grad_enabled(self.learnable):
            for step in range(self.num_steps):
                t_curr = dt * float(step)
                t_mid = t_curr + 0.5 * dt
                t_next = dt * float(step + 1)

                # Evaluate temperature at t_curr, t_mid, t_next
                T_curr = torch.where(t_curr <= ramp_s, T0_K + hr_K_per_s * t_curr, T_peak_K)
                T_mid = torch.where(t_mid <= ramp_s, T0_K + hr_K_per_s * t_mid, T_peak_K)
                T_next = torch.where(t_next <= ramp_s, T0_K + hr_K_per_s * t_next, T_peak_K)

                # RK4 Stage 1
                k1 = self._reaction_rate(alpha, T_curr)
                # RK4 Stage 2
                a_mid1 = torch.clamp(alpha + 0.5 * dt * k1, 0.0, 1.0)
                k2 = self._reaction_rate(a_mid1, T_mid)
                # RK4 Stage 3
                a_mid2 = torch.clamp(alpha + 0.5 * dt * k2, 0.0, 1.0)
                k3 = self._reaction_rate(a_mid2, T_mid)
                # RK4 Stage 4
                a_end = torch.clamp(alpha + dt * k3, 0.0, 1.0)
                k4 = self._reaction_rate(a_end, T_next)

                # Update alpha
                d_alpha = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                alpha = torch.clamp(alpha + d_alpha, 0.0, 1.0)

            # Compute char yield:
            # Each component solid residue = (1 - alpha_i) + alpha_i * char_inf_i
            comp_residue = (1.0 - alpha) + alpha * torch.clamp(self.char_inf.unsqueeze(0), 0.0, 1.0)
            # Weight by organic fractions
            organic_char_fraction = torch.sum(biopolymer_weights * comp_residue, dim=-1, keepdim=True)

            if ash_weight is not None:
                # Ash is conserved in the solid char residue
                # Total char yield = (1 - ash_weight) * organic_char_fraction + ash_weight
                ash_clean = torch.nan_to_num(ash_weight, nan=0.0)
                ash_safe = torch.clamp(ash_clean, 0.0, 0.9)
                total_char_fraction = (1.0 - ash_safe) * organic_char_fraction + ash_safe
            else:
                total_char_fraction = organic_char_fraction

            char_yield_pct = total_char_fraction * 100.0
            return char_yield_pct, alpha

    def _reaction_rate(self, alpha: torch.Tensor, T_K: torch.Tensor) -> torch.Tensor:
        """Computes d(alpha)/dt = k_i(T) * (1 - alpha)^n_i."""
        k = self.get_rate_constant(T_K)  # (Batch, 4)
        rem = torch.clamp(1.0 - alpha, min=0.0)
        n = torch.clamp(self.n_order.unsqueeze(0), min=0.5, max=5.0)
        return k * torch.pow(rem, n)


def solve_kinetic_ode(
    biopolymers: torch.Tensor,
    T_peak: torch.Tensor,
    RT: torch.Tensor,
    HR: torch.Tensor,
    ash: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convenience functional call for stateless kinetic yield evaluation."""
    solver = KineticDecompositionODE(learnable=False)
    char_yield_pct, _ = solver(biopolymers, T_peak, RT, HR, ash_weight=ash)
    return char_yield_pct
