"""
project/utils/energy_physics.py
First-principles thermodynamic energy balance for pyrolysis reactors.
Calculates net specific external energy required (kJ / kg feedstock).
Stateless and physics-derived; requires no empirical training energy data.
"""

from typing import Union
import numpy as np
import torch


def compute_net_energy_balance(
    T_peak_C: Union[float, np.ndarray, torch.Tensor],
    moisture_fraction: Union[float, np.ndarray, torch.Tensor] = 0.0,
    char_yield_wt_pct: Union[float, np.ndarray, torch.Tensor] = 30.0,
    T0_C: float = 25.0,
    Cp_biomass: float = 1.60,         # kJ / (kg * K)
    Cp_water: float = 4.184,          # kJ / (kg * K)
    Cp_steam: float = 2.080,          # kJ / (kg * K)
    Delta_H_vap: float = 2260.0,      # kJ / kg water at 100 °C
    Delta_H_reaction: float = 100.0,  # Endothermic primary devolatilization enthalpy (kJ / kg)
    LHV_syngas: float = 11000.0,      # Lower heating value of non-condensable pyrolysis gas (kJ / kg)
    heat_rec_efficiency: float = 0.50 # Syngas combustion heat recovery efficiency (fraction)
) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Computes net thermal energy required per kg of dry biomass input:
    E_net = E_sensible + E_evap + Delta_H_reaction - E_recovered_syngas

    If E_net < 0, the reaction is autothermal under optimal heat recuperation.
    """
    is_torch = isinstance(T_peak_C, torch.Tensor)

    delta_T = T_peak_C - T0_C
    if is_torch:
        delta_T = torch.clamp(delta_T, min=0.0)
        moist = torch.clamp(moisture_fraction, min=0.0, max=0.8)
        char_frac = torch.clamp(char_yield_wt_pct / 100.0, min=0.0, max=1.0)
    else:
        delta_T = np.maximum(delta_T, 0.0)
        moist = np.clip(moisture_fraction, 0.0, 0.8)
        char_frac = np.clip(char_yield_wt_pct / 100.0, 0.0, 1.0)

    # 1. Sensible heat to raise dry biomass from T0 to T_peak
    E_sensible = Cp_biomass * delta_T

    # 2. Water evaporation energy if moisture > 0
    delta_T_pre_boil = 100.0 - T0_C
    delta_T_superheat = delta_T - delta_T_pre_boil
    if is_torch:
        delta_T_superheat = torch.clamp(delta_T_superheat, min=0.0)
    else:
        delta_T_superheat = np.maximum(delta_T_superheat, 0.0)

    E_evap = moist * (Cp_water * delta_T_pre_boil + Delta_H_vap + Cp_steam * delta_T_superheat)

    # 3. Primary pyrolysis devolatilization enthalpy
    E_reaction = Delta_H_reaction

    # 4. Syngas energy recovery (approx. 25-35% of volatile fraction is non-condensable syngas)
    volatile_frac = 1.0 - char_frac
    syngas_yield_est = volatile_frac * 0.30
    E_syngas_potential = syngas_yield_est * LHV_syngas
    E_recovered = E_syngas_potential * heat_rec_efficiency

    # Total Net Energy Required
    E_net = E_sensible + E_evap + E_reaction - E_recovered
    return E_net
