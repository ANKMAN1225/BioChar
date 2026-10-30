"""
project/real_life/data/dataloader.py
Data loading, validation, and preprocessing for the Real-Life track.
Ensures zero data leakage: scalers are fit strictly on the training partition.
Uses compositional transforms (CLR / ILR) to preserve simplex constraints.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path
# Add project root to sys.path if not present
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.transforms import clr_transform, ilr_transform


def find_column(df: pd.DataFrame, candidates: list) -> str:
    """Finds the matching column from a candidate list, case-insensitive."""
    col_map = {col.lower().strip(): col for col in df.columns}
    for cand in candidates:
        cand_lower = cand.lower().strip()
        if cand_lower in col_map:
            return col_map[cand_lower]
        # Partial match
        for k, v in col_map.items():
            if cand_lower in k:
                return v
    raise KeyError(f"None of candidate columns {candidates} found in dataframe columns: {list(df.columns)}")


class RealLifeDataset(Dataset):
    """PyTorch Dataset for real-life production biochar prediction."""

    def __init__(
        self,
        feedstock_x: np.ndarray,
        process_x: np.ndarray,
        raw_biopolymer_weights: np.ndarray,
        yield_y: np.ndarray,
        feedstock_names: np.ndarray,
        ash_weights: Optional[np.ndarray] = None,
    ):
        self.feedstock_x = torch.tensor(feedstock_x, dtype=torch.float32)
        self.process_x = torch.tensor(process_x, dtype=torch.float32)
        self.raw_biopolymer_weights = torch.tensor(raw_biopolymer_weights, dtype=torch.float32)
        self.yield_y = torch.tensor(yield_y, dtype=torch.float32)
        self.ash_weights = torch.tensor(ash_weights, dtype=torch.float32) if ash_weights is not None else torch.zeros((len(yield_y), 1), dtype=torch.float32)
        self.feedstock_names = feedstock_names

    def __len__(self) -> int:
        return len(self.yield_y)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "feedstock_x": self.feedstock_x[idx],
            "process_x": self.process_x[idx],
            "raw_biopolymers": self.raw_biopolymer_weights[idx],
            "ash_weight": self.ash_weights[idx],
            "yield": self.yield_y[idx],
        }


def load_real_life_data(
    config: dict,
    split_feedstock_held_out: Optional[list] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Dataset, Dataset, Dict[str, StandardScaler]]:
    """
    Loads and preprocesses real-life cohort data.
    Scalers are strictly fitted on training rows only.
    """
    raw_path = config["data"]["raw_path"]
    df = pd.read_csv(raw_path)

    # Filter to model included rows
    if "model_include" in df.columns:
        df = df[df["model_include"] == True].copy()

    # Identify process columns
    t_col = find_column(df, ["T (C)", "T (°C)", "Temperature"])
    rt_col = find_column(df, ["RT (min)", "RT", "Residence_Time"])
    hr_col = find_column(df, ["HR (C/min)", "HR (°C/min)", "Heating_Rate"])
    yield_col = find_column(df, ["Yield-char (%)", "Yield-char", "Yield"])
    feedstock_col = find_column(df, ["biomass type", "feedstock"])
    ash_col = find_column(df, ["Ash (%)", "Ash"])

    mode = config["data"].get("feature_mode", "biopolymers")
    transform_type = config["data"].get("composition_transform", "clr")

    if mode == "biopolymers":
        cel_col = find_column(df, ["Cel (%)", "Cellulose"])
        hem_col = find_column(df, ["Hem (%)", "Hemicellulose"])
        lig_col = find_column(df, ["Lig (%)", "Lignin"])
        ext_col = find_column(df, ["Ext (%)", "Extractives"])
        comp_cols = [cel_col, hem_col, lig_col, ext_col]
        # Drop rows missing any biopolymer component
        valid_mask = df[comp_cols + [t_col, rt_col, hr_col, yield_col]].notna().all(axis=1)
        df_valid = df[valid_mask].copy().reset_index(drop=True)
    elif mode == "proximate":
        vm_col = find_column(df, ["VM (%)", "VM", "Volatile_Matter"])
        fc_col = find_column(df, ["FC (%)", "FC", "Fixed_Carbon"])
        comp_cols = [ash_col, vm_col, fc_col]
        valid_mask = df[comp_cols + [t_col, rt_col, hr_col, yield_col]].notna().all(axis=1)
        df_valid = df[valid_mask].copy().reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported feature_mode: {mode}")

    print(f"[{mode.upper()} MODE] Valid rows available: {len(df_valid)} (out of {len(df)} total).")

    # Grouped split by feedstock or DOI to prevent train/test data leakage
    feedstocks = df_valid[feedstock_col].values
    unique_feedstocks = np.unique(feedstocks)

    np.random.seed(random_state)
    if split_feedstock_held_out is not None:
        test_feedstocks = split_feedstock_held_out
    else:
        n_test = max(1, int(len(unique_feedstocks) * test_size))
        test_feedstocks = np.random.choice(unique_feedstocks, size=n_test, replace=False)

    train_mask = ~np.isin(feedstocks, test_feedstocks)
    test_mask = np.isin(feedstocks, test_feedstocks)

    print(f"Train samples: {np.sum(train_mask)} ({len(unique_feedstocks) - len(test_feedstocks)} feedstocks)")
    print(f"Test samples:  {np.sum(test_mask)} ({len(test_feedstocks)} held-out feedstocks)")

    # Extract composition
    comp_raw = df_valid[comp_cols].values.astype(np.float64)
    # Normalize rows to simplex sum = 1.0
    comp_simplex = comp_raw / np.sum(comp_raw, axis=1, keepdims=True)

    # Apply compositional log-ratio transform
    if transform_type == "clr":
        feedstock_features = clr_transform(comp_simplex)
    elif transform_type == "ilr":
        feedstock_features = ilr_transform(comp_simplex)
    else:
        feedstock_features = comp_simplex

    # Raw biopolymers for kinetic ODE (if mode is proximate, initialize equal proxy fractions)
    if mode == "biopolymers":
        raw_bio = comp_simplex  # (Hemi, Cel, Lig, Ext)
    else:
        # Proxy equal distribution prior for biopolymers when only proximate is known
        raw_bio = np.ones((len(df_valid), 4), dtype=np.float64) * 0.25

    # Ash fraction (if unmeasured in biopolymer study, default to standard organic ash 2.0 wt%)
    ash_raw = df_valid[ash_col].values.astype(np.float64).reshape(-1, 1) / 100.0
    ash_clean = np.nan_to_num(ash_raw, nan=0.02)

    # Process features: [T, RT, HR]
    process_raw = df_valid[[t_col, rt_col, hr_col]].values.astype(np.float64)
    yield_raw = df_valid[yield_col].values.astype(np.float64).reshape(-1, 1)

    # Strictly fit scalers on training split ONLY!
    process_scaler = StandardScaler()
    process_scaler.fit(process_raw[train_mask])
    process_scaled = process_scaler.transform(process_raw)

    yield_scaler = StandardScaler()
    yield_scaler.fit(yield_raw[train_mask])

    scalers = {
        "process": process_scaler,
        "yield": yield_scaler,
    }

    train_dataset = RealLifeDataset(
        feedstock_x=feedstock_features[train_mask],
        process_x=process_scaled[train_mask],
        raw_biopolymer_weights=raw_bio[train_mask],
        yield_y=yield_raw[train_mask],
        feedstock_names=feedstocks[train_mask],
        ash_weights=ash_clean[train_mask],
    )

    test_dataset = RealLifeDataset(
        feedstock_x=feedstock_features[test_mask],
        process_x=process_scaled[test_mask],
        raw_biopolymer_weights=raw_bio[test_mask],
        yield_y=yield_raw[test_mask],
        feedstock_names=feedstocks[test_mask],
        ash_weights=ash_clean[test_mask],
    )

    return train_dataset, test_dataset, scalers
