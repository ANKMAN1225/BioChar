"""
project/utils/__init__.py
Stateless utilities for thermodynamics, ODE kinetics, compositional transforms, and metrics.
No learnable parameters exist in this module.
"""

from .transforms import clr_transform, ilr_transform, inverse_clr, ilr_basis_matrix
from .thermal_profile import build_thermal_profile, ThermalProfile
from .ode_integrator import KineticDecompositionODE, solve_kinetic_ode
from .energy_physics import compute_net_energy_balance
from .metrics import regression_metrics, uq_calibration_metrics

__all__ = [
    "clr_transform",
    "ilr_transform",
    "inverse_clr",
    "ilr_basis_matrix",
    "build_thermal_profile",
    "ThermalProfile",
    "KineticDecompositionODE",
    "solve_kinetic_ode",
    "compute_net_energy_balance",
    "regression_metrics",
    "uq_calibration_metrics",
]
