"""
project/evaluation/run_benchmarks.py
Unified benchmark runner comparing the Physics-Informed Deep Ensemble against:
- Random Forest
- XGBoost
- Gaussian Process Regressor
- MC Dropout MLP
Evaluated on the exact same held-out feedstock splits with identical metrics.
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from real_life.data.dataloader import load_real_life_data
from real_life.train import train_single_model
from real_life.model.network import RealLifePINN
from evaluation.baselines import train_and_eval_baselines
from utils.metrics import regression_metrics, uq_calibration_metrics


def run_full_benchmark():
    cfg_file = PROJECT_ROOT / "real_life" / "config.yaml"
    with open(cfg_file, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Running Comprehensive Cross-Model Benchmark ===")

    train_dataset, test_dataset, scalers = load_real_life_data(config)
    save_dir = PROJECT_ROOT / "real_life" / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train PINN Deep Ensemble
    ensemble_size = 5
    models = []
    print(f"\nTraining PINN Deep Ensemble (M={ensemble_size})...")
    for m in range(ensemble_size):
        m_model = train_single_model(
            model_idx=m,
            config=config,
            train_dataset=train_dataset,
            scalers=scalers,
            device=device,
            save_dir=save_dir,
        )
        models.append(m_model)

    # PINN Ensemble inference on test set
    f_x_te = test_dataset.feedstock_x.to(device)
    p_x_te = test_dataset.process_x.to(device)
    raw_bio_te = test_dataset.raw_biopolymer_weights.to(device)
    ash_te = test_dataset.ash_weights.to(device)
    y_test = test_dataset.yield_y.cpu().numpy().ravel()

    raw_proc_np = scalers["process"].inverse_transform(p_x_te.cpu().numpy())
    raw_proc_te = torch.tensor(raw_proc_np, dtype=torch.float32, device=device)

    all_means = []
    all_stds = []
    for m_model in models:
        m_model.eval()
        with torch.no_grad():
            out = m_model(
                feedstock_x=f_x_te,
                process_x=p_x_te,
                raw_process=raw_proc_te,
                raw_biopolymers=raw_bio_te,
                ash_weight=ash_te,
            )
            all_means.append(out["yield_mean"].cpu().numpy().ravel())
            all_stds.append(out["yield_std"].cpu().numpy().ravel())

    means_arr = np.array(all_means)
    stds_arr = np.array(all_stds)

    pinn_mean = np.mean(means_arr, axis=0)
    pinn_epistemic = np.std(means_arr, axis=0)
    pinn_aleatoric = np.mean(stds_arr, axis=0)
    pinn_total_std = np.sqrt(pinn_epistemic ** 2 + pinn_aleatoric ** 2)

    pinn_reg = regression_metrics(y_test, pinn_mean)
    pinn_uq = uq_calibration_metrics(y_test, pinn_mean, pinn_total_std)

    benchmark_rows = [{
        "Model": "PINN Deep Ensemble (Ours)",
        "RMSE (wt%)": pinn_reg["rmse"],
        "MAE (wt%)": pinn_reg["mae"],
        "R^2": pinn_reg["r2"],
        "95% Coverage (PICP)": pinn_uq["picp"],
        "MPIW (wt%)": pinn_uq["mpiw"],
        "Gaussian NLL": pinn_uq["nll"],
    }]

    # 2. Prepare flat features for tabular baselines [Feedstock_features, Process_scaled]
    X_train_flat = np.hstack([
        train_dataset.feedstock_x.cpu().numpy(),
        train_dataset.process_x.cpu().numpy(),
    ])
    y_train_flat = train_dataset.yield_y.cpu().numpy().ravel()

    X_test_flat = np.hstack([
        test_dataset.feedstock_x.cpu().numpy(),
        test_dataset.process_x.cpu().numpy(),
    ])

    # 3. Benchmark standard baselines
    print("\nTraining and evaluating baseline models...")
    baseline_res = train_and_eval_baselines(X_train_flat, y_train_flat, X_test_flat, y_test)

    for name, met in baseline_res.items():
        benchmark_rows.append({
            "Model": name,
            "RMSE (wt%)": met["rmse"],
            "MAE (wt%)": met["mae"],
            "R^2": met["r2"],
            "95% Coverage (PICP)": met.get("picp", float("nan")),
            "MPIW (wt%)": met.get("mpiw", float("nan")),
            "Gaussian NLL": met.get("nll", float("nan")),
        })

    df_bench = pd.DataFrame(benchmark_rows)
    print("\n" + "=" * 80)
    print("                     CROSS-MODEL BENCHMARK RESULTS")
    print("=" * 80)
    print(df_bench.to_string(index=False))

    out_csv = PROJECT_ROOT / "evaluation" / "benchmark_comparison.csv"
    df_bench.to_csv(out_csv, index=False)
    print(f"\nSaved benchmark results to: {out_csv}")
    return df_bench


if __name__ == "__main__":
    run_full_benchmark()
