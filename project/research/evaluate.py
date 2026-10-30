"""
project/research/evaluate.py
Evaluates Research track Deep Ensemble on multi-target properties (Yield, N-char, SSA-char)
and runs Multi-Objective Pareto active learning recommendations.
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

from research.data.dataloader import load_research_data
from research.model.network import ResearchPINN
from research.acquisition.optimizer import recommend_research_experiments
from utils.metrics import regression_metrics, uq_calibration_metrics


def load_research_ensemble(config: dict, feedstock_dim: int, device: torch.device):
    ckpt_dir = PROJECT_ROOT / "research" / "checkpoints"
    ensemble_size = config["model"].get("ensemble_size", 5)

    models = []
    for m in range(ensemble_size):
        model = ResearchPINN(
            feedstock_dim=feedstock_dim,
            process_dim=3,
            num_targets=3,
            feedstock_hidden=tuple(config["model"]["feedstock_encoder"]["hidden_dims"]),
            process_hidden=tuple(config["model"]["process_encoder"]["hidden_dims"]),
            fusion_hidden=tuple(config["model"]["fusion_head"]["hidden_dims"]),
            dropout=0.0,
            learnable_kinetics=False,
        ).to(device)

        ckpt_file = ckpt_dir / f"research_member_{m}.pt"
        if not ckpt_file.exists():
            raise FileNotFoundError(f"Missing checkpoint file: {ckpt_file}. Run research/train.py first.")
        model.load_state_dict(torch.load(ckpt_file, map_location=device))
        model.eval()
        models.append(model)

    return models


def evaluate_research():
    cfg_file = PROJECT_ROOT / "research" / "config.yaml"
    with open(cfg_file, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Research Track: Evaluation & Multi-Objective Recommendations ===")

    train_dataset, test_dataset, scalers = load_research_data(config)
    feedstock_dim = train_dataset.feedstock_x.shape[1]

    models = load_research_ensemble(config, feedstock_dim=feedstock_dim, device=device)
    print(f"Successfully loaded {len(models)} Research PINN ensemble members.")

    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    test_batch = next(iter(test_loader))

    f_x = test_batch["feedstock_x"].to(device)
    p_x = test_batch["process_x"].to(device)
    raw_bio = test_batch["raw_biopolymers"].to(device)
    ash = test_batch["ash_weight"].to(device)
    targets_true = test_batch["targets"].cpu().numpy()
    target_mask = test_batch["target_mask"].cpu().numpy()
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
            all_means.append(out["means"].cpu().numpy())
            all_stds.append(out["stds"].cpu().numpy())

    means_arr = np.array(all_means) # (M, N, 3)
    stds_arr = np.array(all_stds)

    y_pred_mean = np.mean(means_arr, axis=0) # (N, 3)
    sigma_epistemic = np.std(means_arr, axis=0)
    sigma_aleatoric = np.mean(stds_arr, axis=0)
    sigma_total = np.sqrt(sigma_epistemic ** 2 + sigma_aleatoric ** 2)

    target_names = ["Yield-char (wt%)", "N-char (wt%)", "SSA-char (m2/g)"]

    print("\n--- Multi-Target Performance on Held-Out Test Feedstocks ---")
    for t_idx, name in enumerate(target_names):
        valid_idx = target_mask[:, t_idx] > 0
        if np.sum(valid_idx) == 0:
            continue
        yt = targets_true[valid_idx, t_idx]
        yp = y_pred_mean[valid_idx, t_idx]
        ys = sigma_total[valid_idx, t_idx]

        reg = regression_metrics(yt, yp)
        uq = uq_calibration_metrics(yt, yp, ys)
        print(f"\nTarget [{name}] (n={np.sum(valid_idx)}):")
        print(f"  RMSE: {reg['rmse']:.3f} | MAE: {reg['mae']:.3f} | R^2: {reg['r2']:.3f}")
        print(f"  95% CI Coverage (PICP): {uq['picp'] * 100:.1f}% | MPIW: {uq['mpiw']:.3f} | NLL: {uq['nll']:.3f}")

    # Generate multi-objective Pareto recommendations for sample test feedstock
    unique_feedstocks = np.unique(feedstocks)
    sample_feedstock = unique_feedstocks[0]
    sample_idx = np.where(feedstocks == sample_feedstock)[0][0]

    sample_f_feat = test_dataset.feedstock_x[sample_idx].cpu().numpy()
    sample_raw_bio = test_dataset.raw_biopolymers[sample_idx].cpu().numpy()
    sample_ash = float(test_dataset.ash_weights[sample_idx].cpu().numpy().item())

    print(f"\n--- Multi-Objective Pareto Recommendations for Target: '{sample_feedstock}' ---")
    recs = recommend_research_experiments(
        ensemble_models=models,
        feedstock_features=sample_f_feat,
        process_scaler=scalers["process"],
        config=config,
        raw_biopolymers=sample_raw_bio,
        ash_weight=sample_ash,
    )
    rec_df = pd.DataFrame(recs)
    print(rec_df.to_string(index=False))

    return recs


if __name__ == "__main__":
    evaluate_research()
