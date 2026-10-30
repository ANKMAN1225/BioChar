"""
project/tests/test_all_components.py
Unit and integration tests for transforms, thermal history, kinetic ODE, energy physics,
masked losses, and acquisition functions.
"""

import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.transforms import clr_transform, inverse_clr, ilr_transform, inverse_ilr
from utils.thermal_profile import build_thermal_profile
from utils.ode_integrator import KineticDecompositionODE, solve_kinetic_ode
from utils.energy_physics import compute_net_energy_balance
from real_life.model.losses import HeteroscedasticLoss, PhysicsInformedLoss
from research.model.losses import MaskedHeteroscedasticLoss
from real_life.acquisition.optimizer import CostAwareBatchUCB


def test_compositional_transforms():
    """Verifies CLR and ILR invertibility and simplex preservation."""
    # 4-part composition (e.g. Cel, Hem, Lig, Ext)
    x = np.array([[0.40, 0.25, 0.30, 0.05], [0.50, 0.20, 0.20, 0.10]])

    # CLR
    clr = clr_transform(x)
    assert clr.shape == (2, 4)
    # Sum of CLR components should be zero (on hyperplane)
    assert np.allclose(np.sum(clr, axis=1), 0.0, atol=1e-5)
    rec_clr = inverse_clr(clr)
    assert np.allclose(x, rec_clr, atol=1e-5)

    # ILR
    ilr = ilr_transform(x)
    assert ilr.shape == (2, 3) # D - 1 coordinates
    rec_ilr = inverse_ilr(ilr)
    assert np.allclose(x, rec_ilr, atol=1e-5)
    print("[PASS] Compositional transforms passed invertibility and hyperplane checks.")


def test_thermal_profile():
    """Verifies piecewise ramp and hold behavior."""
    profile = build_thermal_profile(T_peak_C=500.0, residence_time_min=30.0, heating_rate_C_per_min=10.0, T0_C=25.0)
    # Ramp time = (500 - 25) / 10 = 47.5 min = 2850 s
    assert np.isclose(profile.ramp_duration_s, 2850.0)
    # Check start temp
    assert np.isclose(profile.temperature_at_t(0.0), 25.0 + 273.15)
    # Check peak hold temp
    assert np.isclose(profile.temperature_at_t(3000.0), 500.0 + 273.15)
    print("[PASS] Thermal profile piecewise trajectory passed.")


def test_kinetic_ode_differentiability():
    """Verifies kinetic ODE layer yields reasonable yield bounds and supports autograd."""
    bio = torch.tensor([[0.25, 0.40, 0.30, 0.05]], dtype=torch.float32)
    T = torch.tensor([[500.0]], dtype=torch.float32, requires_grad=True)
    RT = torch.tensor([[30.0]], dtype=torch.float32)
    HR = torch.tensor([[10.0]], dtype=torch.float32)

    layer = KineticDecompositionODE(learnable=False)
    yield_pct, alpha = layer(bio, T, RT, HR)

    # Check bounds
    assert 0.0 <= yield_pct.item() <= 100.0
    # Check gradient flow with respect to temperature
    yield_pct.backward()
    assert T.grad is not None
    # As temperature increases, char yield decreases: d(Yield)/dT <= 0
    assert T.grad.item() <= 0.0
    print("[PASS] Kinetic ODE gradient and physical monotonicity checks passed.")


def test_net_energy_balance():
    """Verifies sensible heat and syngas recovery behavior."""
    # Low temp
    e_low = compute_net_energy_balance(T_peak_C=300.0, char_yield_wt_pct=45.0)
    # High temp
    e_high = compute_net_energy_balance(T_peak_C=800.0, char_yield_wt_pct=20.0)
    # Higher temperature requires more thermal input
    assert e_high > e_low
    print("[PASS] Thermodynamic energy balance scaling passed.")


def test_masked_heteroscedastic_loss():
    """Verifies that missing targets with mask=0 do not contribute to gradients."""
    means = torch.tensor([[10.0, 20.0], [15.0, 25.0]], requires_grad=True)
    log_vars = torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)
    targets = torch.tensor([[10.0, 999.0], [15.0, 999.0]]) # Second column has garbage/missing
    mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]]) # Second column masked out

    criterion = MaskedHeteroscedasticLoss()
    loss = criterion(means, log_vars, targets, mask)
    loss.backward()

    # Gradients for masked column should be strictly zero!
    assert torch.all(means.grad[:, 1] == 0.0)
    assert torch.all(log_vars.grad[:, 1] == 0.0)
    print("[PASS] Masked heteroscedastic loss successfully masks missing targets.")


def test_acquisition_optimizer_bounds():
    """Verifies that acquisition proposals lie strictly within defined physical bounds."""
    bounds = {
        "Temperature": (350.0, 750.0),
        "Residence_Time": (15.0, 120.0),
        "Heating_Rate": (5.0, 30.0),
    }
    opt = CostAwareBatchUCB(bounds=bounds, batch_size=3, num_candidates=500)
    cands = opt.sample_candidate_grid()

    assert np.all(cands[:, 0] >= 350.0) and np.all(cands[:, 0] <= 750.0)
    assert np.all(cands[:, 1] >= 15.0) and np.all(cands[:, 1] <= 120.0)
    assert np.all(cands[:, 2] >= 5.0) and np.all(cands[:, 2] <= 30.0)
    print("[PASS] Acquisition bounds verification passed.")


if __name__ == "__main__":
    test_compositional_transforms()
    test_thermal_profile()
    test_kinetic_ode_differentiability()
    test_net_energy_balance()
    test_masked_heteroscedastic_loss()
    test_acquisition_optimizer_bounds()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
