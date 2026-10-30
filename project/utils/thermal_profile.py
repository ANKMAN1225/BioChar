"""
project/utils/thermal_profile.py
Explicit thermal history profile T(t) for pyrolysis.
Parameterizes linear ramp at heating rate HR followed by residence hold at peak temperature T_peak.
All thermodynamic calculations use Kelvin internally.
"""

from dataclasses import dataclass
from typing import Tuple, Union
import numpy as np
import torch


@dataclass(frozen=True)
class ThermalProfile:
    T0_K: float
    T_peak_K: float
    heating_rate_K_per_s: float
    residence_time_s: float
    ramp_duration_s: float
    total_duration_s: float

    def temperature_at_t(self, t_s: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """Calculates temperature T(t) in Kelvin at elapsed time t_s in seconds."""
        if isinstance(t_s, torch.Tensor):
            ramp_temp = self.T0_K + self.heating_rate_K_per_s * t_s
            is_ramp = t_s <= self.ramp_duration_s
            return torch.where(is_ramp, ramp_temp, torch.tensor(self.T_peak_K, dtype=t_s.dtype, device=t_s.device))
        else:
            t = np.asarray(t_s, dtype=np.float64)
            ramp_temp = self.T0_K + self.heating_rate_K_per_s * t
            return np.where(t <= self.ramp_duration_s, ramp_temp, self.T_peak_K)


def build_thermal_profile(
    T_peak_C: float,
    residence_time_min: float,
    heating_rate_C_per_min: float,
    T0_C: float = 25.0
) -> ThermalProfile:
    """
    Constructs a ThermalProfile from standard process units:
    - T_peak_C: peak pyrolysis temperature in Celsius
    - residence_time_min: hold duration in minutes
    - heating_rate_C_per_min: heating rate in °C/min
    - T0_C: initial ambient temperature in Celsius (default 25.0 °C)
    """
    T0_K = float(T0_C + 273.15)
    T_peak_K = float(T_peak_C + 273.15)
    
    # Enforce physical constraints
    if T_peak_K < T0_K:
        T_peak_K = T0_K
    hr_min = max(float(heating_rate_C_per_min), 0.1)
    rt_min = max(float(residence_time_min), 0.0)

    # Convert to SI units (K/s and seconds)
    hr_K_per_s = hr_min / 60.0
    rt_s = rt_min * 60.0
    ramp_duration_s = (T_peak_K - T0_K) / hr_K_per_s
    total_duration_s = ramp_duration_s + rt_s

    return ThermalProfile(
        T0_K=T0_K,
        T_peak_K=T_peak_K,
        heating_rate_K_per_s=hr_K_per_s,
        residence_time_s=rt_s,
        ramp_duration_s=ramp_duration_s,
        total_duration_s=total_duration_s,
    )


def discretize_thermal_profile(
    profile: ThermalProfile,
    num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates discrete time grid (seconds) and temperature grid (Kelvin)."""
    t_grid = np.linspace(0.0, profile.total_duration_s, num_points)
    T_grid = profile.temperature_at_t(t_grid)
    return t_grid, T_grid
