"""
project/research/acquisition/optimizer.py
Multi-Objective Pareto & Information Gain Acquisition for the Research track.
Identifies non-dominated Pareto trade-offs across:
- Maximizing biochar yield
- Maximizing surface area (SSA) / carbon quality
- Maximizing epistemic information gain across unobserved chemical domains.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn


def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """
    Finds Pareto-efficient points for minimization: costs of shape (N, D).
    Returns boolean mask of length N.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point with a lower cost in at least one objective
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True  # Self is not dominated by self
    return is_efficient


class MultiObjectiveParetoAcquisition:
    """
    Multi-objective acquisition identifying the Pareto frontier of biochar properties.
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        batch_size: int = 4,
        num_candidates: int = 4000,
    ):
        self.bounds = bounds
        self.batch_size = batch_size
        self.num_candidates = num_candidates

    def sample_candidates(self, random_state: int = 42) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        t_low, t_high = self.bounds["Temperature"]
        rt_low, rt_high = self.bounds["Residence_Time"]
        hr_low, hr_high = self.bounds["Heating_Rate"]

        T = rng.uniform(t_low, t_high, size=(self.num_candidates, 1))
        RT = rng.uniform(rt_low, rt_high, size=(self.num_candidates, 1))
        HR = rng.uniform(hr_low, hr_high, size=(self.num_candidates, 1))
        return np.hstack([T, RT, HR])

    def evaluate_candidates(
        self,
        ensemble_models: List[nn.Module],
        feedstock_features: np.ndarray,
        process_scaler,
        candidates_raw: np.ndarray,
        raw_biopolymers: np.ndarray,
        ash_weight: float = 0.05,
    ) -> Dict[str, np.ndarray]:
        device = next(ensemble_models[0].parameters()).device
        N = len(candidates_raw)

        f_t = torch.tensor(np.tile(feedstock_features, (N, 1)), dtype=torch.float32, device=device)
        p_scaled = process_scaler.transform(candidates_raw)
        p_t = torch.tensor(p_scaled, dtype=torch.float32, device=device)
        raw_p_t = torch.tensor(candidates_raw, dtype=torch.float32, device=device)
        raw_bio_t = torch.tensor(np.tile(raw_biopolymers, (N, 1)), dtype=torch.float32, device=device)
        ash_t = torch.tensor(np.full((N, 1), ash_weight), dtype=torch.float32, device=device)

        ensemble_preds = []
        for model in ensemble_models:
            model.eval()
            with torch.no_grad():
                out = model(
                    feedstock_x=f_t,
                    process_x=p_t,
                    raw_process=raw_p_t,
                    raw_biopolymers=raw_bio_t,
                    ash_weight=ash_t,
                )
                ensemble_preds.append(out["means"].cpu().numpy())

        # Shape (M, N, num_targets)
        preds_arr = np.array(ensemble_preds)
        means = np.mean(preds_arr, axis=0)      # (N, num_targets)
        stds = np.std(preds_arr, axis=0)        # (N, num_targets)

        # Objective: Maximize Yield (col 0) and Maximize SSA (col 2) with exploration bonus
        ucb_yield = means[:, 0] + 1.0 * stds[:, 0]
        ucb_ssa = means[:, 2] + 1.0 * stds[:, 2]

        # For Pareto minimization, invert objectives
        costs = np.column_stack([-ucb_yield, -ucb_ssa])
        pareto_mask = is_pareto_efficient(costs)

        return {
            "candidates": candidates_raw,
            "means": means,
            "stds": stds,
            "pareto_mask": pareto_mask,
            "ucb_yield": ucb_yield,
            "ucb_ssa": ucb_ssa,
        }

    def select_batch(self, eval_results: Dict[str, np.ndarray]) -> List[Dict[str, float]]:
        pareto_indices = np.where(eval_results["pareto_mask"])[0]
        candidates = eval_results["candidates"][pareto_indices]
        means = eval_results["means"][pareto_indices]
        stds = eval_results["stds"][pareto_indices]

        # Select diverse points across the Pareto frontier
        if len(pareto_indices) <= self.batch_size:
            selected_idx = list(range(len(pareto_indices)))
        else:
            # Sort by predicted yield and pick evenly spaced points across frontier
            sort_order = np.argsort(means[:, 0])
            spaced_idx = np.linspace(0, len(sort_order) - 1, self.batch_size, dtype=int)
            selected_idx = sort_order[spaced_idx]

        results = []
        for i in selected_idx:
            results.append({
                "Temperature_C": float(candidates[i, 0]),
                "Residence_Time_min": float(candidates[i, 1]),
                "Heating_Rate_C_per_min": float(candidates[i, 2]),
                "Expected_Yield_pct": float(means[i, 0]),
                "Expected_SSA_m2_g": float(means[i, 2]),
                "Yield_Uncertainty": float(stds[i, 0]),
                "SSA_Uncertainty": float(stds[i, 2]),
            })
        return results


def recommend_research_experiments(
    ensemble_models: List[nn.Module],
    feedstock_features: np.ndarray,
    process_scaler,
    config: dict,
    raw_biopolymers: np.ndarray,
    ash_weight: float = 0.05,
) -> List[Dict[str, float]]:
    acq_cfg = config["acquisition"]
    optimizer = MultiObjectiveParetoAcquisition(
        bounds=acq_cfg["bounds"],
        batch_size=acq_cfg.get("batch_size", 4),
    )
    candidates = optimizer.sample_candidates()
    eval_res = optimizer.evaluate_candidates(
        ensemble_models=ensemble_models,
        feedstock_features=feedstock_features,
        process_scaler=process_scaler,
        candidates_raw=candidates,
        raw_biopolymers=raw_biopolymers,
        ash_weight=ash_weight,
    )
    return optimizer.select_batch(eval_res)
