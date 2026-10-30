"""
project/real_life/evaluate.py
Evaluation and closed-loop recommendation script for the Real-Life track.
Loads trained checkpoints, assesses held-out generalization, computes energy balances,
and generates cost-aware active learning recommendations.
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from real_life.data.dataloader import load_real_life_data
from real_life.model.network import RealLifePINN
from real_life.acquisition.optimizer import recommend_next_experiments
from utils.energy_physics import compute_net_energy_balance
from utils.metrics import regression_metrics, uq_calibration_metrics


def load_trained_ensemble(config: dict, feedstock_dim: int, device: torch.device):
    """Loads all saved checkpoint members into memory."""
    ckpt_dir = PROJECT_ROOT / "real_life" / "checkpoints"
    ensemble_size = config["model"].get("ensemble_size", 5)
    mode = config["data"].get("feature_mode", "biopolymers")

    models = []
    for m in range(ensemble_size):
        model = RealLifePINN(
            feedstock_dim=feedstock_dim,
            process_dim=3,
            feedstock_hidden=tuple(config["model"]["feedstock_encoder"]["hidden_dims"]),
            process_hidden=tuple(config["model"]["process_encoder"]["hidden_dims"]),
            fusion_hidden=tuple(config["model"]["fusion_head"]["hidden_dims"]),
            dropout=0.0,
            mode=mode,
        ).to(device)

        ckpt_file = ckpt_dir / f"pinn_member_{m}.pt"
        if not ckpt_file.exists():
            raise FileNotFoundError(f"Missing checkpoint file: {ckpt_file}. Run train.py first.")
        model.load_state_dict(torch.load(ckpt_file, map_location=device))
        model.eval()
        models.append(model)

    return models


def evaluate_and_recommend():
    cfg_file = PROJECT_ROOT / "real_life" / "config.yaml"
    with open(cfg_file, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Real-Life Track: Evaluation & Closed-Loop Recommendation ===")

    train_dataset, test_dataset, scalers = load_real_life_data(config)
    feedstock_dim = train_dataset.feedstock_x.shape[1]

    models = load_trained_ensemble(config, feedstock_dim=feedstock_dim, device=device)
    print(f"Successfully loaded {len(models)} PINN ensemble members.")

    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    test_batch = next(iter(test_loader))

    f_x = test_batch["feedstock_x"].to(device)
    p_x = test_batch["process_x"].to(device)
    raw_bio = test_batch["raw_biopolymers"].to(device)
    ash = test_batch["ash_weight"].to(device)
    y_true = test_batch["yield"].cpu().numpy().ravel()
    feedstocks = test_dataset.feedstock_names

    raw_proc_np = scalers["process"].inverse_transform(p_x.cpu().numpy())
    raw_proc = torch.tensor(raw_proc_np, dtype=torch.float32, device=device)

    all_means = []
    all_stds = []
    for model in models:
        with torch.no_grad():
            out = model(
                feedstock_x=f_x,
                process_x=p_x,
                raw_process=raw_proc,
                raw_biopolymers=raw_bio,
                ash_weight=ash,
            )
            all_means.append(out["yield_mean"].cpu().numpy().ravel())
            all_stds.append(out["yield_std"].cpu().numpy().ravel())

    means_arr = np.array(all_means)
    stds_arr = np.array(all_stds)

    y_pred_mean = np.mean(means_arr, axis=0)
    sigma_epistemic = np.std(means_arr, axis=0)
    sigma_aleatoric = np.mean(stds_arr, axis=0)
    sigma_total = np.sqrt(sigma_epistemic ** 2 + sigma_aleatoric ** 2)

    # Net energy balance for observed test points
    t_peak_arr = raw_proc_np[:, 0]
    net_energy_arr = compute_net_energy_balance(
        T_peak_C=t_peak_arr,
        moisture_fraction=0.0,
        char_yield_wt_pct=y_pred_mean,
    )

    reg_metrics = regression_metrics(y_true, y_pred_mean)
    uq_metrics = uq_calibration_metrics(y_true, y_pred_mean, sigma_total)

    print("\n--- Model Generalization Metrics on Held-Out Test Feedstocks ---")
    print(f"  RMSE: {reg_metrics['rmse']:.3f} wt% | MAE: {reg_metrics['mae']:.3f} wt% | R^2: {reg_metrics['r2']:.3f}")
    print(f"  95% CI Coverage (PICP): {uq_metrics['picp'] * 100:.1f}% | MPIW: {uq_metrics['mpiw']:.3f} wt% | NLL: {uq_metrics['nll']:.3f}")
    print(f"  Average specific energy requirement: {np.mean(net_energy_arr):.1f} kJ / kg dry biomass")

    # Sample an example held-out feedstock and generate batch recommendations
    unique_test_feedstocks = np.unique(feedstocks)
    sample_feedstock = unique_test_feedstocks[0]
    sample_idx = np.where(feedstocks == sample_feedstock)[0][0]

    sample_f_feat = test_dataset.feedstock_x[sample_idx].cpu().numpy()
    sample_raw_bio = test_dataset.raw_biopolymer_weights[sample_idx].cpu().numpy()
    sample_ash = float(test_dataset.ash_weights[sample_idx].cpu().numpy().item())

    print(f"\n--- Closed-Loop Active Learning Recommendations for Target: '{sample_feedstock}' ---")
    recommendations = recommend_next_experiments(
        ensemble_models=models,
        feedstock_features=sample_f_feat,
        process_scaler=scalers["process"],
        config=config,
        raw_biopolymers=sample_raw_bio,
        ash_weight=sample_ash,
    )

    rec_df = pd.DataFrame(recommendations)
    print(rec_df.to_string(index=False))

    return reg_metrics, uq_metrics, recommendations


if __name__ == "__main__":
    evaluate_and_recommend()
