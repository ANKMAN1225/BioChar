"""
project/sandbox/test_proximate_mode.py
Sandbox validation of Real-Life Track in "Proximate Analysis" fallback mode (Block 2).
Verifies that using cheap Ash, Volatile Matter, and Fixed Carbon with the learned
biopolymer projection layer works seamlessly without code rewrites.
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
from real_life.model.network import RealLifePINN
from real_life.model.losses import PhysicsInformedLoss
from real_life.train import sample_collocation_points
from utils.metrics import regression_metrics, uq_calibration_metrics


def run_proximate_mode_test():
    print("=" * 80)
    print("      SANDBOX TEST: REAL-LIFE TRACK IN PROXIMATE FALLBACK MODE (BLOCK 2)")
    print("=" * 80)

    # Temporary sandbox config with feature_mode: proximate
    prox_cfg = {
        "track": "real_life_proximate",
        "seed": 42,
        "data": {
            "raw_path": "cleaned_data/leng_yield_cohort.csv",
            "basis": "dry",
            "feature_mode": "proximate",
            "composition_transform": "clr",
        },
        "model": {
            "feedstock_encoder": {"hidden_dims": [64, 32], "latent_dim": 16, "dropout": 0.1},
            "process_encoder": {"hidden_dims": [32, 16], "latent_dim": 8, "dropout": 0.1},
            "fusion_head": {"hidden_dims": [64, 32]},
        },
        "kinetics": {"num_steps": 30},
        "training": {"epochs": 120, "batch_size": 32, "lr": 1e-3, "weight_decay": 1e-4, "adaptive_weights": True},
    }

    train_ds, test_ds, scalers = load_real_life_data(prox_cfg)
    device = torch.device("cpu")

    feedstock_dim = train_ds.feedstock_x.shape[1]
    print(f"\nFeedstock Input Dimension (Proximate CLR): {feedstock_dim}")

    model = RealLifePINN(
        feedstock_dim=feedstock_dim,
        process_dim=3,
        feedstock_hidden=(64, 32),
        process_hidden=(32, 16),
        fusion_hidden=(64, 32),
        mode="proximate",
    ).to(device)

    loss_fn = PhysicsInformedLoss(adaptive=True).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(loss_fn.parameters()), lr=1e-3, weight_decay=1e-4)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

    print("\nTraining PINN with Learned Biopolymer Surrogate Mapper...")
    for epoch in range(120):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            f_x = batch["feedstock_x"]
            p_x = batch["process_x"]
            raw_bio = batch["raw_biopolymers"]
            ash = batch["ash_weight"]
            y_true = batch["yield"]

            raw_proc = torch.tensor(scalers["process"].inverse_transform(p_x.numpy()), dtype=torch.float32)

            pred_dict = model(f_x, p_x, raw_proc, raw_bio, ash, compute_kinetics=True)

            # Collocation
            n_c = min(len(f_x), 16)
            T_c, p_c_scaled, p_c_raw = sample_collocation_points(
                batch_size=n_c,
                feedstock_features=f_x[:n_c],
                raw_biopolymers=raw_bio[:n_c],
                ash_weight=ash[:n_c],
                process_scaler=scalers["process"],
                device=device,
            )
            colloc_out = model(f_x[:n_c], p_c_scaled, p_c_raw, raw_bio[:n_c], ash[:n_c], compute_kinetics=False)
            grad_T = torch.autograd.grad(
                outputs=colloc_out["yield_mean"],
                inputs=T_c,
                grad_outputs=torch.ones_like(colloc_out["yield_mean"]),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            loss, d = loss_fn(pred_dict, y_true, collocation_grad_T=grad_T)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 30 == 0:
            print(f"  [Epoch {epoch + 1:3d}/120] Loss: {d['total_loss']:.4f} (Data: {d['loss_data']:.3f}, Kin: {d['loss_kinetics']:.3f})")

    # Evaluate on held-out test feedstocks
    model.eval()
    with torch.no_grad():
        test_f_x = test_ds.feedstock_x
        test_p_x = test_ds.process_x
        test_bio = test_ds.raw_biopolymer_weights
        test_ash = test_ds.ash_weights
        test_y = test_ds.yield_y.numpy().ravel()

        test_raw_proc = torch.tensor(scalers["process"].inverse_transform(test_p_x.numpy()), dtype=torch.float32)
        out = model(test_f_x, test_p_x, test_raw_proc, test_bio, test_ash, compute_kinetics=False)

        pred_y = out["yield_mean"].numpy().ravel()
        pred_std = out["yield_std"].numpy().ravel()

    reg = regression_metrics(test_y, pred_y)
    uq = uq_calibration_metrics(test_y, pred_y, pred_std)

    print("\n--- Proximate Fallback Generalization Performance ---")
    print(f"  MAE:  {reg['mae']:.3f} wt%")
    print(f"  RMSE: {reg['rmse']:.3f} wt%")
    print(f"  R^2:  {reg['r2']:.3f}")
    print(f"  95% CI Coverage (PICP): {uq['picp'] * 100:.1f}%")

    res_df = pd.DataFrame([{
        "Input Mode": "Proximate Fallback (Ash, VM, FC)",
        "MAE (wt%)": reg["mae"],
        "RMSE (wt%)": reg["rmse"],
        "R^2": reg["r2"],
        "95% Coverage (PICP)": uq["picp"],
    }])
    res_csv = PROJECT_ROOT / "sandbox" / "sandbox_proximate_mode_metrics.csv"
    res_df.to_csv(res_csv, index=False)
    print(f"\nSaved proximate mode benchmark to: {res_csv}")


if __name__ == "__main__":
    run_proximate_mode_test()
