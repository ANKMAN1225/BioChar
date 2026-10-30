"""
project/research/data/dataloader.py
Data loader and preprocessing for the Research track.
Integrates all feature blocks (Biopolymers, Proximate, Ultimate, Process)
and constructs multi-target masks for partially observed properties (Yield, N-char, SSA-char).
Prevents data leakage: scalers are fit strictly on the training partition.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.transforms import clr_transform, ilr_transform
from real_life.data.dataloader import find_column


class ResearchDataset(Dataset):
    """PyTorch Dataset for multi-target research biochar modeling with missing value masks."""

    def __init__(
        self,
        feedstock_x: np.ndarray,
        process_x: np.ndarray,
        raw_biopolymers: np.ndarray,
        targets_y: np.ndarray,
        target_masks: np.ndarray,
        feedstock_names: np.ndarray,
        ash_weights: np.ndarray,
    ):
        self.feedstock_x = torch.tensor(feedstock_x, dtype=torch.float32)
        self.process_x = torch.tensor(process_x, dtype=torch.float32)
        self.raw_biopolymers = torch.tensor(raw_biopolymers, dtype=torch.float32)
        self.targets_y = torch.tensor(targets_y, dtype=torch.float32)
        self.target_masks = torch.tensor(target_masks, dtype=torch.float32)
        self.ash_weights = torch.tensor(ash_weights, dtype=torch.float32)
        self.feedstock_names = feedstock_names

    def __len__(self) -> int:
        return len(self.targets_y)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "feedstock_x": self.feedstock_x[idx],
            "process_x": self.process_x[idx],
            "raw_biopolymers": self.raw_biopolymers[idx],
            "targets": self.targets_y[idx],
            "target_mask": self.target_masks[idx],
            "ash_weight": self.ash_weights[idx],
        }


def load_research_data(
    config: dict,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Dataset, Dataset, Dict[str, StandardScaler]]:
    """
    Loads research dataset with all feature blocks and multi-target masks.
    """
    raw_path = config["data"]["raw_path"]
    df = pd.read_csv(raw_path)

    if "model_include" in df.columns:
        df = df[df["model_include"] == True].copy()

    # Locate feature columns
    cel_col = find_column(df, ["Cel (%)", "Cellulose"])
    hem_col = find_column(df, ["Hem (%)", "Hemicellulose"])
    lig_col = find_column(df, ["Lig (%)", "Lignin"])
    ext_col = find_column(df, ["Ext (%)", "Extractives"])
    ash_col = find_column(df, ["Ash (%)", "Ash"])
    vm_col = find_column(df, ["VM (%)", "VM"])
    fc_col = find_column(df, ["FC (%)", "FC"])
    c_col = find_column(df, ["C (%)", "C"])
    h_col = find_column(df, ["H (%)", "H"])
    n_col = find_column(df, ["N (%)", "N"])
    o_col = find_column(df, ["O (%)", "O"])

    t_col = find_column(df, ["T (C)", "T (°C)", "Temperature", "T ("])
    rt_col = find_column(df, ["RT (min)", "Residence_Time", "RT"])
    hr_col = find_column(df, ["HR (C/min)", "Heating_Rate", "HR (", "HR"])

    # Locate targets
    yield_col = find_column(df, ["Yield-char (%)", "Yield"])
    n_char_col = find_column(df, ["N-char (%)", "N-char"])
    ssa_col = find_column(df, ["SSA-char (m2/g)", "SSA"])
    feedstock_col = find_column(df, ["biomass type", "feedstock"])

    # Require complete features in Block 1, Proximate, Ultimate, and Process
    feature_cols = [cel_col, hem_col, lig_col, ext_col, ash_col, vm_col, fc_col, c_col, h_col, n_col, o_col, t_col, rt_col, hr_col]
    valid_mask = df[feature_cols].notna().all(axis=1) & df[yield_col].notna()
    df_valid = df[valid_mask].copy().reset_index(drop=True)

    print(f"[RESEARCH TRACK] Valid complete-feature samples: {len(df_valid)} (out of {len(df)})")

    feedstocks = df_valid[feedstock_col].values
    unique_feedstocks = np.unique(feedstocks)

    # Held-out feedstock split
    np.random.seed(random_state)
    n_test = max(1, int(len(unique_feedstocks) * test_size))
    test_feedstocks = np.random.choice(unique_feedstocks, size=n_test, replace=False)

    train_mask = ~np.isin(feedstocks, test_feedstocks)
    test_mask = np.isin(feedstocks, test_feedstocks)

    print(f"Train samples: {np.sum(train_mask)} ({len(unique_feedstocks) - len(test_feedstocks)} feedstocks)")
    print(f"Test samples:  {np.sum(test_mask)} ({len(test_feedstocks)} held-out feedstocks)")

    # 1. Biopolymers sub-composition
    bio_raw = df_valid[[cel_col, hem_col, lig_col, ext_col]].values.astype(np.float64)
    bio_simplex = bio_raw / np.sum(bio_raw, axis=1, keepdims=True)

    # 2. Proximate sub-composition
    prox_raw = df_valid[[ash_col, vm_col, fc_col]].values.astype(np.float64)
    prox_simplex = prox_raw / np.sum(prox_raw, axis=1, keepdims=True)

    # 3. Ultimate sub-composition
    ult_raw = df_valid[[c_col, h_col, n_col, o_col]].values.astype(np.float64)
    ult_simplex = ult_raw / np.sum(ult_raw, axis=1, keepdims=True)

    # Transform sub-compositions to unconstrained coordinates via ILR
    transform_name = config["data"].get("composition_transform", "ilr")
    if transform_name == "ilr":
        bio_t = ilr_transform(bio_simplex)     # shape (N, 3)
        prox_t = ilr_transform(prox_simplex)   # shape (N, 2)
        ult_t = ilr_transform(ult_simplex)     # shape (N, 3)
    else:
        bio_t = clr_transform(bio_simplex)
        prox_t = clr_transform(prox_simplex)
        ult_t = clr_transform(ult_simplex)

    # Combined feedstock representation (8 features)
    feedstock_features = np.hstack([bio_t, prox_t, ult_t])

    # Process features [T, RT, HR]
    process_raw = df_valid[[t_col, rt_col, hr_col]].values.astype(np.float64)
    ash_raw = df_valid[ash_col].values.astype(np.float64).reshape(-1, 1) / 100.0

    # Multi-targets: [Yield, N-char, SSA-char]
    y_yield = df_valid[yield_col].values.astype(np.float64)
    y_n_char = df_valid[n_char_col].values.astype(np.float64) if n_char_col in df_valid else np.full_like(y_yield, np.nan)
    y_ssa = df_valid[ssa_col].values.astype(np.float64) if ssa_col in df_valid else np.full_like(y_yield, np.nan)

    raw_targets = np.column_stack([y_yield, y_n_char, y_ssa])
    target_masks = (~np.isnan(raw_targets)).astype(np.float64)
    # Fill NaN targets with 0.0 for tensor construction (loss is zeroed out by mask)
    clean_targets = np.nan_to_num(raw_targets, nan=0.0)

    # Fit scalers strictly on training split
    process_scaler = StandardScaler()
    process_scaler.fit(process_raw[train_mask])
    process_scaled = process_scaler.transform(process_raw)

    scalers = {"process": process_scaler}

    train_dataset = ResearchDataset(
        feedstock_x=feedstock_features[train_mask],
        process_x=process_scaled[train_mask],
        raw_biopolymers=bio_simplex[train_mask],
        targets_y=clean_targets[train_mask],
        target_masks=target_masks[train_mask],
        feedstock_names=feedstocks[train_mask],
        ash_weights=ash_raw[train_mask],
    )

    test_dataset = ResearchDataset(
        feedstock_x=feedstock_features[test_mask],
        process_x=process_scaled[test_mask],
        raw_biopolymers=bio_simplex[test_mask],
        targets_y=clean_targets[test_mask],
        target_masks=target_masks[test_mask],
        feedstock_names=feedstocks[test_mask],
        ash_weights=ash_raw[test_mask],
    )

    return train_dataset, test_dataset, scalers
