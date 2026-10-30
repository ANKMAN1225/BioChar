"""
project/real_life/train.py
Trains a Deep Ensemble of Physics-Informed Neural Networks for the Real-Life track.
Implements:
- Strict zero-leakage training/test splits grouped by feedstock.
- Deep Ensemble (M >= 5 members) with independent seeds.
- Differentiable kinetic ODE consistency loss.
- Physics collocation points enforcing d(Yield)/dT <= 0.
- Kendall & Gal adaptive uncertainty loss weighting.
- Checkpoint persistence.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from real_life.data.dataloader import load_real_life_data
from real_life.model.network import RealLifePINN
from real_life.model.losses import PhysicsInformedLoss
from utils.metrics import regression_metrics, uq_calibration_metrics


def sample_collocation_points(
    batch_size: int,
    feedstock_features: torch.Tensor,
    raw_biopolymers: torch.Tensor,
    ash_weight: torch.Tensor,
    process_scaler,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Samples synthetic collocation process conditions across physical ranges
    to enforce thermodynamic monotonicity d(Yield)/dT <= 0.
    """
    T_c = torch.empty(batch_size, 1, device=device).uniform_(300.0, 900.0)
    RT_c = torch.empty(batch_size, 1, device=device).uniform_(10.0, 180.0)
    HR_c = torch.empty(batch_size, 1, device=device).uniform_(5.0, 50.0)
    T_c.requires_grad_(True)

    # Scale process conditions using train-fit scaler
    # Since T_c needs gradient tracking, perform affine scaling differentiably:
    t_mean = float(process_scaler.mean_[0])
    t_scale = float(process_scaler.scale_[0])
    rt_mean = float(process_scaler.mean_[1])
    rt_scale = float(process_scaler.scale_[1])
    hr_mean = float(process_scaler.mean_[2])
    hr_scale = float(process_scaler.scale_[2])

    T_scaled = (T_c - t_mean) / t_scale
    RT_scaled = (RT_c - rt_mean) / rt_scale
    HR_scaled = (HR_c - hr_mean) / hr_scale

    process_scaled = torch.cat([T_scaled, RT_scaled, HR_scaled], dim=-1)
    raw_process = torch.cat([T_c, RT_c, HR_c], dim=-1)

    return T_c, process_scaled, raw_process


def train_single_model(
    model_idx: int,
    config: dict,
    train_dataset,
    scalers,
    device: torch.device,
    save_dir: Path,
) -> RealLifePINN:
    """Trains a single PINN ensemble member."""
    seed = config.get("seed", 42) + model_idx * 101
    torch.manual_seed(seed)
    np.random.seed(seed)

    batch_size = config["training"].get("batch_size", 32)
    epochs = config["training"].get("epochs", 250)
    lr = float(config["training"].get("lr", 1e-3))
    weight_decay = float(config["training"].get("weight_decay", 1e-4))
    adaptive = config["training"].get("adaptive_weights", True)
    mode = config["data"].get("feature_mode", "biopolymers")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    feedstock_dim = train_dataset.feedstock_x.shape[1]
    model = RealLifePINN(
        feedstock_dim=feedstock_dim,
        process_dim=3,
        feedstock_hidden=tuple(config["model"]["feedstock_encoder"]["hidden_dims"]),
        process_hidden=tuple(config["model"]["process_encoder"]["hidden_dims"]),
        fusion_hidden=tuple(config["model"]["fusion_head"]["hidden_dims"]),
        dropout=config["model"]["feedstock_encoder"].get("dropout", 0.1),
        mode=mode,
    ).to(device)

    loss_fn = PhysicsInformedLoss(adaptive=adaptive).to(device)

    optimizer = optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    process_scaler = scalers["process"]

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            optimizer.zero_grad()

            f_x = batch["feedstock_x"].to(device)
            p_x = batch["process_x"].to(device)
            raw_bio = batch["raw_biopolymers"].to(device)
            ash = batch["ash_weight"].to(device)
            y_true = batch["yield"].to(device)

            # Unscale process variables for kinetic ODE
            raw_proc_np = process_scaler.inverse_transform(p_x.cpu().numpy())
            raw_proc = torch.tensor(raw_proc_np, dtype=torch.float32, device=device)

            pred_dict = model(
                feedstock_x=f_x,
                process_x=p_x,
                raw_process=raw_proc,
                raw_biopolymers=raw_bio,
                ash_weight=ash,
            )

            # Collocation points for monotonicity: d(Yield)/dT <= 0
            n_colloc = min(len(f_x), 16)
            T_c, p_c_scaled, p_c_raw = sample_collocation_points(
                batch_size=n_colloc,
                feedstock_features=f_x[:n_colloc],
                raw_biopolymers=raw_bio[:n_colloc],
                ash_weight=ash[:n_colloc],
                process_scaler=process_scaler,
                device=device,
            )

            colloc_out = model(
                feedstock_x=f_x[:n_colloc],
                process_x=p_c_scaled,
                raw_process=p_c_raw,
                raw_biopolymers=raw_bio[:n_colloc],
                ash_weight=ash[:n_colloc],
                compute_kinetics=False,
            )
            colloc_yield = colloc_out["yield_mean"]

            # Compute gradient d(yield)/dT
            grad_T = torch.autograd.grad(
                outputs=colloc_yield,
                inputs=T_c,
                grad_outputs=torch.ones_like(colloc_yield),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            loss, loss_dict = loss_fn(pred_dict, y_true, collocation_grad_T=grad_T)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            epoch_losses.append(loss_dict["total_loss"])

        scheduler.step()

        if (epoch + 1) % 25 == 0 or epoch == epochs - 1:
            mean_loss = np.mean(epoch_losses)
            print(f"  [Model {model_idx + 1} | Epoch {epoch + 1:3d}/{epochs}] Loss: {mean_loss:.4f} "
                  f"(Data: {loss_dict['loss_data']:.3f}, Kin: {loss_dict['loss_kinetics']:.3f}, Mono: {loss_dict['loss_mono']:.3f})", flush=True)

    # Save model checkpoint
    save_path = save_dir / f"pinn_member_{model_idx}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"  -> Saved checkpoint: {save_path.name}", flush=True)

    return model


def run_ensemble_training(config_path: str = "real_life/config.yaml"):
    """Main training entry point for Deep Ensemble on real-life track."""
    cfg_file = PROJECT_ROOT / config_path
    with open(cfg_file, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Real-Life PINN Training on device: {device} ===")
    print(f"Feature mode: {config['data'].get('feature_mode')} | Transform: {config['data'].get('composition_transform')}")

    train_dataset, test_dataset, scalers = load_real_life_data(config)

    save_dir = PROJECT_ROOT / "real_life" / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    ensemble_size = config["model"].get("ensemble_size", 5)
    print(f"\nTraining Deep Ensemble with M = {ensemble_size} members...")

    models = []
    for m in range(ensemble_size):
        print(f"\n--- Training Ensemble Member {m + 1}/{ensemble_size} ---")
        model = train_single_model(
            model_idx=m,
            config=config,
            train_dataset=train_dataset,
            scalers=scalers,
            device=device,
            save_dir=save_dir,
        )
        models.append(model)

    # Evaluate Deep Ensemble on held-out test feedstocks
    print("\n=== Evaluating Deep Ensemble on Held-Out Test Feedstocks ===")
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    test_batch = next(iter(test_loader))

    f_x = test_batch["feedstock_x"].to(device)
    p_x = test_batch["process_x"].to(device)
    raw_bio = test_batch["raw_biopolymers"].to(device)
    ash = test_batch["ash_weight"].to(device)
    y_true = test_batch["yield"].cpu().numpy().ravel()

    raw_proc_np = scalers["process"].inverse_transform(p_x.cpu().numpy())
    raw_proc = torch.tensor(raw_proc_np, dtype=torch.float32, device=device)

    all_means = []
    all_stds = []

    for model in models:
        model.eval()
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

    means_arr = np.array(all_means) # (M, N)
    stds_arr = np.array(all_stds)

    y_pred_mean = np.mean(means_arr, axis=0)
    sigma_epistemic = np.std(means_arr, axis=0)
    sigma_aleatoric = np.mean(stds_arr, axis=0)
    sigma_total = np.sqrt(sigma_epistemic ** 2 + sigma_aleatoric ** 2)

    # Compute regression & UQ calibration metrics
    reg_metrics = regression_metrics(y_true, y_pred_mean)
    uq_metrics = uq_calibration_metrics(y_true, y_pred_mean, sigma_total, nominal_confidence=0.95)

    print(f"Test Set Performance (Held-out Feedstocks):")
    print(f"  RMSE: {reg_metrics['rmse']:.3f} wt%")
    print(f"  MAE:  {reg_metrics['mae']:.3f} wt%")
    print(f"  R^2:  {reg_metrics['r2']:.3f}")
    print(f"Uncertainty Quantification (95% CI):")
    print(f"  Coverage (PICP): {uq_metrics['picp'] * 100:.1f}% (Nominal: 95.0%)")
    print(f"  Mean Interval Width (MPIW): {uq_metrics['mpiw']:.3f} wt%")
    print(f"  Gaussian NLL: {uq_metrics['nll']:.3f}")
    print(f"  Mean Epistemic Uncertainty: {np.mean(sigma_epistemic):.3f}")
    print(f"  Mean Aleatoric Uncertainty: {np.mean(sigma_aleatoric):.3f}")

    return models, scalers, reg_metrics, uq_metrics


if __name__ == "__main__":
    run_ensemble_training()
