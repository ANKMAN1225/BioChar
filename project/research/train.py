"""
project/research/train.py
Training pipeline for the Research track Deep Ensemble.
Integrates all feature blocks, multi-target masked supervision, learnable kinetic ODE,
and physics collocation constraints.
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from research.data.dataloader import load_research_data
from research.model.network import ResearchPINN
from research.model.losses import ResearchPhysicsLoss
from real_life.train import sample_collocation_points
from utils.metrics import regression_metrics, uq_calibration_metrics


def train_research_model(
    model_idx: int,
    config: dict,
    train_dataset,
    scalers,
    device: torch.device,
    save_dir: Path,
) -> ResearchPINN:
    seed = config.get("seed", 42) + model_idx * 211
    torch.manual_seed(seed)
    np.random.seed(seed)

    batch_size = config["training"].get("batch_size", 32)
    epochs = config["training"].get("epochs", 300)
    lr = float(config["training"].get("lr", 8e-4))
    weight_decay = float(config["training"].get("weight_decay", 1e-4))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    feedstock_dim = train_dataset.feedstock_x.shape[1]

    model = ResearchPINN(
        feedstock_dim=feedstock_dim,
        process_dim=3,
        num_targets=3,
        feedstock_hidden=tuple(config["model"]["feedstock_encoder"]["hidden_dims"]),
        process_hidden=tuple(config["model"]["process_encoder"]["hidden_dims"]),
        fusion_hidden=tuple(config["model"]["fusion_head"]["hidden_dims"]),
        dropout=config["model"]["feedstock_encoder"].get("dropout", 0.1),
        learnable_kinetics=config["kinetics"].get("learnable", True),
    ).to(device)

    loss_fn = ResearchPhysicsLoss(adaptive=config["training"].get("adaptive_weights", True)).to(device)

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
            targets = batch["targets"].to(device)
            mask = batch["target_mask"].to(device)
            ash = batch["ash_weight"].to(device)

            raw_proc_np = process_scaler.inverse_transform(p_x.cpu().numpy())
            raw_proc = torch.tensor(raw_proc_np, dtype=torch.float32, device=device)

            pred_dict = model(
                feedstock_x=f_x,
                process_x=p_x,
                raw_process=raw_proc,
                raw_biopolymers=raw_bio,
                ash_weight=ash,
            )

            # Collocation points for monotonicity on Yield: d(Yield)/dT <= 0
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
            colloc_yield = colloc_out["means"][:, 0:1]

            grad_T = torch.autograd.grad(
                outputs=colloc_yield,
                inputs=T_c,
                grad_outputs=torch.ones_like(colloc_yield),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            loss, loss_dict = loss_fn(pred_dict, targets, mask, collocation_grad_T=grad_T)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            epoch_losses.append(loss_dict["total_loss"])

        scheduler.step()

        if (epoch + 1) % 25 == 0 or epoch == epochs - 1:
            mean_loss = np.mean(epoch_losses)
            print(f"  [Research Model {model_idx + 1} | Epoch {epoch + 1:3d}/{epochs}] Loss: {mean_loss:.4f} "
                  f"(Data: {loss_dict['loss_data']:.3f}, Kin: {loss_dict['loss_kinetics']:.3f}, Mono: {loss_dict['loss_mono']:.3f})", flush=True)

    save_path = save_dir / f"research_member_{model_idx}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"  -> Saved checkpoint: {save_path.name}", flush=True)
    return model


def run_research_training(config_path: str = "research/config.yaml"):
    cfg_file = PROJECT_ROOT / config_path
    with open(cfg_file, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Research Track PINN Training on device: {device} ===")

    train_dataset, test_dataset, scalers = load_research_data(config)
    save_dir = PROJECT_ROOT / "research" / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    ensemble_size = config["model"].get("ensemble_size", 5)
    print(f"\nTraining Deep Ensemble with M = {ensemble_size} members...")

    models = []
    for m in range(ensemble_size):
        print(f"\n--- Training Research Ensemble Member {m + 1}/{ensemble_size} ---")
        model = train_research_model(
            model_idx=m,
            config=config,
            train_dataset=train_dataset,
            scalers=scalers,
            device=device,
            save_dir=save_dir,
        )
        models.append(model)

    print("\n=== Research Track Ensemble Training Complete ===")
    return models, scalers


if __name__ == "__main__":
    run_research_training()
