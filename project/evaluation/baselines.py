"""
project/evaluation/baselines.py
Standard machine learning baseline models evaluated on the exact same held-out splits:
1. Random Forest Regressor
2. XGBoost Regressor
3. Gaussian Process Regressor (GPR with RBF + WhiteKernel)
4. Standard MLP with Monte Carlo Dropout (MCDO) for uncertainty quantification
"""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from xgboost import XGBRegressor

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.metrics import regression_metrics, uq_calibration_metrics


class MCDropoutMLP(nn.Module):
    """Standard Multilayer Perceptron with Monte Carlo Dropout."""

    def __init__(self, in_dim: int, hidden_dims: Tuple[int, ...] = (64, 32), dropout: float = 0.2):
        super().__init__()
        layers = []
        curr = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr = h
        layers.append(nn.Linear(curr, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_mc(self, x: torch.Tensor, num_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Runs multiple forward passes with active dropout to quantify epistemic uncertainty."""
        self.train() # Keep dropout active during evaluation
        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                preds.append(self.forward(x).cpu().numpy().ravel())
        preds_arr = np.array(preds)
        mean = np.mean(preds_arr, axis=0)
        std = np.std(preds_arr, axis=0)
        return mean, std


def train_and_eval_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Trains and benchmarks all baseline models."""
    results = {}

    # 1. Random Forest
    rf = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=random_state)
    rf.fit(X_train, y_train.ravel())
    # Empirical tree variance for UQ
    tree_preds = np.array([tree.predict(X_test) for tree in rf.estimators_])
    rf_mean = np.mean(tree_preds, axis=0)
    rf_std = np.std(tree_preds, axis=0)
    results["Random Forest"] = {
        **regression_metrics(y_test, rf_mean),
        **uq_calibration_metrics(y_test, rf_mean, rf_std),
    }

    # 2. XGBoost
    xgb = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=random_state)
    xgb.fit(X_train, y_train.ravel())
    xgb_pred = xgb.predict(X_test)
    results["XGBoost"] = {
        **regression_metrics(y_test, xgb_pred),
        "picp": float("nan"),
        "mpiw": float("nan"),
        "nll": float("nan"),
    }

    # 3. Gaussian Process Regressor
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, random_state=random_state, alpha=1e-2)
    gpr.fit(X_train, y_train.ravel())
    gp_mean, gp_std = gpr.predict(X_test, return_std=True)
    results["Gaussian Process"] = {
        **regression_metrics(y_test, gp_mean),
        **uq_calibration_metrics(y_test, gp_mean, gp_std),
    }

    # 4. MC Dropout MLP
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_te_t = torch.tensor(X_test, dtype=torch.float32)

    mlp = MCDropoutMLP(in_dim=X_train.shape[1], hidden_dims=(64, 32), dropout=0.2)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.005, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    mlp.train()
    for _ in range(200):
        optimizer.zero_grad()
        out = mlp(X_tr_t)
        loss = loss_fn(out, y_tr_t)
        loss.backward()
        optimizer.step()

    mc_mean, mc_std = mlp.predict_mc(X_te_t, num_samples=50)
    results["MC Dropout MLP"] = {
        **regression_metrics(y_test, mc_mean),
        **uq_calibration_metrics(y_test, mc_mean, mc_std),
    }

    return results
