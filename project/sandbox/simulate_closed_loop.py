"""
project/sandbox/simulate_closed_loop.py
Sandbox simulation comparing:
1. Real-Life Track (Cost-aware batch UCB with cheap biopolymer/proximate inputs)
2. Research Track (Multi-objective Pareto optimization with full chemical characterization)
3. Random Search Baseline (Standard trial-and-error in labs)

Runs a 4-round active learning closed-loop simulation on a completely held-out biomass feedstock.
Zero impact on production code/checkpoints.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from real_life.data.dataloader import load_real_life_data
from real_life.model.network import RealLifePINN
from real_life.acquisition.optimizer import recommend_next_experiments
from research.data.dataloader import load_research_data
from research.model.network import ResearchPINN
from research.acquisition.optimizer import recommend_research_experiments
from utils.energy_physics import compute_net_energy_balance
from utils.metrics import regression_metrics


def simulate_lab_experiment(feedstock_name: str, T: float, RT: float, HR: float) -> dict:
    """
    Ground-truth experimental simulator based on empirical Arrhenius devolatilization
    and ash retention for the held-out feedstock.
    """
    # Realistic ground-truth physics with experimental measurement noise (+- 1.5 wt%)
    # Char yield decreases exponentially with temperature
    t_factor = np.exp(-(T - 300.0) / 160.0)
    rt_factor = np.exp(-RT / 300.0)
    base_yield = 18.0 + 32.0 * t_factor * rt_factor
    noise = np.random.normal(0.0, 1.2)
    measured_yield = float(np.clip(base_yield + noise, 12.0, 65.0))

    # SSA increases with temperature up to ~650 C then micro-pores collapse
    if T <= 650.0:
        base_ssa = 5.0 + 0.18 * (T - 300.0) + 0.10 * RT
    else:
        base_ssa = 5.0 + 0.18 * (650.0 - 300.0) - 0.12 * (T - 650.0)
    measured_ssa = float(np.clip(base_ssa + np.random.normal(0.0, 3.0), 2.0, 120.0))

    # Net energy balance
    net_energy = float(compute_net_energy_balance(T_peak_C=T, char_yield_wt_pct=measured_yield))

    return {
        "Yield_pct": measured_yield,
        "SSA_m2_g": measured_ssa,
        "Net_Energy_kJ_kg": net_energy,
    }


def run_sandbox_simulation():
    print("=" * 80)
    print("      SANDBOX SIMULATION: REAL-LIFE TRACK vs. RESEARCH TRACK vs. RANDOM")
    print("=" * 80)

    # 1. Load Real-Life Ensemble
    rl_cfg_file = PROJECT_ROOT / "real_life" / "config.yaml"
    with open(rl_cfg_file, "r") as f:
        rl_cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    rl_train_ds, rl_test_ds, rl_scalers = load_real_life_data(rl_cfg)

    rl_models = []
    rl_ckpt_dir = PROJECT_ROOT / "real_life" / "checkpoints"
    for m in range(5):
        m_model = RealLifePINN(feedstock_dim=rl_train_ds.feedstock_x.shape[1], mode="biopolymers")
        m_model.load_state_dict(torch.load(rl_ckpt_dir / f"pinn_member_{m}.pt", map_location=device))
        m_model.eval()
        rl_models.append(m_model)

    # 2. Load Research Ensemble
    res_cfg_file = PROJECT_ROOT / "research" / "config.yaml"
    with open(res_cfg_file, "r") as f:
        res_cfg = yaml.safe_load(f)

    res_train_ds, res_test_ds, res_scalers = load_research_data(res_cfg)
    res_models = []
    res_ckpt_dir = PROJECT_ROOT / "research" / "checkpoints"
    for m in range(5):
        m_model = ResearchPINN(
            feedstock_dim=res_train_ds.feedstock_x.shape[1],
            num_targets=3,
            learnable_kinetics=False,
        )
        m_model.load_state_dict(torch.load(res_ckpt_dir / f"research_member_{m}.pt", map_location=device))
        m_model.eval()
        res_models.append(m_model)

    # Pick target held-out test biomass
    target_feedstock = str(rl_test_ds.feedstock_names[0])
    print(f"\nTarget Held-Out Biomass Feedstock: '{target_feedstock}'")
    print("Simulating 4 sequential active learning experimental rounds (3 runs/batch = 12 total experiments)...\n")

    f_idx = 0
    rl_f_feat = rl_test_ds.feedstock_x[f_idx].numpy()
    rl_raw_bio = rl_test_ds.raw_biopolymer_weights[f_idx].numpy()
    rl_ash = float(rl_test_ds.ash_weights[f_idx].numpy().item())

    res_f_feat = res_test_ds.feedstock_x[f_idx].numpy()
    res_raw_bio = res_test_ds.raw_biopolymers[f_idx].numpy()
    res_ash = float(res_test_ds.ash_weights[f_idx].numpy().item())

    history_rl = []
    history_res = []
    history_rnd = []

    np.random.seed(42)

    for round_num in range(1, 5):
        print(f"--- ACTIVE LEARNING ROUND {round_num} ---")

        # A. Real-Life Track: Cost-Aware Batch UCB
        rl_recs = recommend_next_experiments(
            ensemble_models=rl_models,
            feedstock_features=rl_f_feat,
            process_scaler=rl_scalers["process"],
            config=rl_cfg,
            raw_biopolymers=rl_raw_bio,
            ash_weight=rl_ash,
        )
        for rec in rl_recs:
            res_lab = simulate_lab_experiment(target_feedstock, rec["Temperature_C"], rec["Residence_Time_min"], rec["Heating_Rate_C_per_min"])
            history_rl.append({
                "Round": round_num,
                "Track": "Real-Life (Cost-Aware PINN)",
                "Temperature_C": rec["Temperature_C"],
                "Residence_Time_min": rec["Residence_Time_min"],
                "Heating_Rate_C_per_min": rec["Heating_Rate_C_per_min"],
                "Measured_Yield_pct": res_lab["Yield_pct"],
                "Measured_SSA_m2_g": res_lab["SSA_m2_g"],
                "Net_Energy_kJ_kg": res_lab["Net_Energy_kJ_kg"],
            })

        # B. Research Track: Multi-Objective Pareto Acquisition
        res_recs = recommend_research_experiments(
            ensemble_models=res_models,
            feedstock_features=res_f_feat,
            process_scaler=res_scalers["process"],
            config=res_cfg,
            raw_biopolymers=res_raw_bio,
            ash_weight=res_ash,
        )
        # Take top 3 recommendations from Pareto batch
        for rec in res_recs[:3]:
            res_lab = simulate_lab_experiment(target_feedstock, rec["Temperature_C"], rec["Residence_Time_min"], rec["Heating_Rate_C_per_min"])
            history_res.append({
                "Round": round_num,
                "Track": "Research (Pareto Multi-Obj PINN)",
                "Temperature_C": rec["Temperature_C"],
                "Residence_Time_min": rec["Residence_Time_min"],
                "Heating_Rate_C_per_min": rec["Heating_Rate_C_per_min"],
                "Measured_Yield_pct": res_lab["Yield_pct"],
                "Measured_SSA_m2_g": res_lab["SSA_m2_g"],
                "Net_Energy_kJ_kg": res_lab["Net_Energy_kJ_kg"],
            })

        # C. Random Trial-and-Error Baseline
        for _ in range(3):
            rnd_t = np.random.uniform(300.0, 900.0)
            rnd_rt = np.random.uniform(10.0, 180.0)
            rnd_hr = np.random.uniform(5.0, 50.0)
            res_lab = simulate_lab_experiment(target_feedstock, rnd_t, rnd_rt, rnd_hr)
            history_rnd.append({
                "Round": round_num,
                "Track": "Random Trial & Error",
                "Temperature_C": rnd_t,
                "Residence_Time_min": rnd_rt,
                "Heating_Rate_C_per_min": rnd_hr,
                "Measured_Yield_pct": res_lab["Yield_pct"],
                "Measured_SSA_m2_g": res_lab["SSA_m2_g"],
                "Net_Energy_kJ_kg": res_lab["Net_Energy_kJ_kg"],
            })

        rl_best_yield = max(h["Measured_Yield_pct"] for h in history_rl)
        res_best_yield = max(h["Measured_Yield_pct"] for h in history_res)
        res_best_ssa = max(h["Measured_SSA_m2_g"] for h in history_res)
        rnd_best_yield = max(h["Measured_Yield_pct"] for h in history_rnd)

        print(f"  [Round {round_num} Summary]")
        print(f"    * Real-Life Track: Best Yield = {rl_best_yield:.2f} wt% | Avg Net Energy = {np.mean([h['Net_Energy_kJ_kg'] for h in history_rl]):.1f} kJ/kg")
        print(f"    * Research Track:  Best Yield = {res_best_yield:.2f} wt% | Best SSA = {res_best_ssa:.1f} m2/g")
        print(f"    * Random Search:   Best Yield = {rnd_best_yield:.2f} wt%")

    # Combine all results into summary dataframe
    all_history = history_rl + history_res + history_rnd
    df_history = pd.DataFrame(all_history)
    out_csv = PROJECT_ROOT / "sandbox" / "sandbox_closed_loop_comparison.csv"
    df_history.to_csv(out_csv, index=False)
    print(f"\nSaved full simulation logs to: {out_csv}")

    # Comparative analysis table
    summary_data = [
        {
            "Strategy": "Real-Life Track (Cost-Aware PINN)",
            "Max Yield Found (wt%)": df_history[df_history["Track"] == "Real-Life (Cost-Aware PINN)"]["Measured_Yield_pct"].max(),
            "Avg Specific Energy (kJ/kg)": df_history[df_history["Track"] == "Real-Life (Cost-Aware PINN)"]["Net_Energy_kJ_kg"].mean(),
            "Avg Furnace Temp (C)": df_history[df_history["Track"] == "Real-Life (Cost-Aware PINN)"]["Temperature_C"].mean(),
            "Primary Focus": "High Yield + Low Energy Cost (Industrial Production)",
        },
        {
            "Strategy": "Research Track (Pareto Multi-Obj)",
            "Max Yield Found (wt%)": df_history[df_history["Track"] == "Research (Pareto Multi-Obj PINN)"]["Measured_Yield_pct"].max(),
            "Avg Specific Energy (kJ/kg)": df_history[df_history["Track"] == "Research (Pareto Multi-Obj PINN)"]["Net_Energy_kJ_kg"].mean(),
            "Avg Furnace Temp (C)": df_history[df_history["Track"] == "Research (Pareto Multi-Obj PINN)"]["Temperature_C"].mean(),
            "Primary Focus": "High Surface Area (SSA) + Carbon Quality Exploration",
        },
        {
            "Strategy": "Random Trial & Error (Lab Baseline)",
            "Max Yield Found (wt%)": df_history[df_history["Track"] == "Random Trial & Error"]["Measured_Yield_pct"].max(),
            "Avg Specific Energy (kJ/kg)": df_history[df_history["Track"] == "Random Trial & Error"]["Net_Energy_kJ_kg"].mean(),
            "Avg Furnace Temp (C)": df_history[df_history["Track"] == "Random Trial & Error"]["Temperature_C"].mean(),
            "Primary Focus": "Unguided / Ad-Hoc Sampling",
        },
    ]
    df_summary = pd.DataFrame(summary_data)
    print("\n" + "=" * 90)
    print("                    FINAL SANDBOX COMPARATIVE PERFORMANCE")
    print("=" * 90)
    print(df_summary.to_string(index=False))

    summary_csv = PROJECT_ROOT / "sandbox" / "sandbox_summary_metrics.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"\nSummary table saved to: {summary_csv}")


if __name__ == "__main__":
    run_sandbox_simulation()
