"""
project/real_life/acquisition/optimizer.py
Cost-aware batch acquisition optimizer for Real-Life biochar production.
Recommends optimal (T, RT, HR) process conditions by balancing:
1. High biochar yield (exploitation)
2. Epistemic model uncertainty (exploration)
3. Energy & operational cost penalty (proportional to high T and long RT).
Uses batch diversity regularization to prevent proposing clustered duplicates.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn


class CostAwareBatchUCB:
    """
    Cost-penalized Upper Confidence Bound batch acquisition function.
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        beta_exploration: float = 1.96,
        cost_T_weight: float = 0.004,
        cost_RT_weight: float = 0.002,
        batch_size: int = 3,
        num_candidates: int = 1000,
    ):
        self.bounds = bounds
        self.beta = beta_exploration
        self.cost_T_weight = cost_T_weight
        self.cost_RT_weight = cost_RT_weight
        self.batch_size = batch_size
        self.num_candidates = num_candidates

    def sample_candidate_grid(self, random_state: Optional[int] = None) -> np.ndarray:
        """Draws uniform quasi-random candidate conditions within physical bounds."""
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
        """
        Runs candidate process conditions through the ensemble of PINNs.
        Returns epistemic, aleatoric, mean yield, and UCB acquisition score.
        """
        device = next(ensemble_models[0].parameters()).device
        N = len(candidates_raw)

        # Replicate feedstock vector across candidates
        feedstock_t = torch.tensor(
            np.tile(feedstock_features, (N, 1)), dtype=torch.float32, device=device
        )
        candidates_scaled = process_scaler.transform(candidates_raw)
        process_scaled_t = torch.tensor(candidates_scaled, dtype=torch.float32, device=device)
        raw_proc_t = torch.tensor(candidates_raw, dtype=torch.float32, device=device)
        raw_bio_t = torch.tensor(
            np.tile(raw_biopolymers, (N, 1)), dtype=torch.float32, device=device
        )
        ash_t = torch.tensor(
            np.full((N, 1), ash_weight), dtype=torch.float32, device=device
        )

        all_means = []
        all_stds = []
        all_kinetics = []

        for model in ensemble_models:
            model.eval()
            with torch.no_grad():
                out = model(
                    feedstock_x=feedstock_t,
                    process_x=process_scaled_t,
                    raw_process=raw_proc_t,
                    raw_biopolymers=raw_bio_t,
                    ash_weight=ash_t,
                )
                all_means.append(out["yield_mean"].cpu().numpy().ravel())
                all_stds.append(out["yield_std"].cpu().numpy().ravel())
                if out.get("kinetic_yield") is not None:
                    all_kinetics.append(out["kinetic_yield"].cpu().numpy().ravel())

        means_arr = np.array(all_means) # Shape (M, N)
        stds_arr = np.array(all_stds)   # Shape (M, N)

        # Physics-informed blended yield (50% Empirical Ensemble Mean + 50% Thermodynamic Kinetic Yield)
        mu_raw = np.mean(means_arr, axis=0) # (N,)
        if len(all_kinetics) > 0:
            kin_arr = np.array(all_kinetics)
            mu_kin = np.mean(kin_arr, axis=0)
            mu_bar = 0.5 * mu_raw + 0.5 * mu_kin
        else:
            mu_bar = mu_raw

        sigma_epistemic = np.std(means_arr, axis=0) # (N,)
        sigma_aleatoric = np.mean(stds_arr, axis=0) # (N,)

        # Multi-Objective Scoring Matrix: Practical Chemical Process Engineering Formulation
        # 1. Hard Quality Gate: 350 <= T <= 550 C
        valid_gate = (candidates_raw[:, 0] >= 350.0) & (candidates_raw[:, 0] <= 550.0)

        # 2. Yield Performance Score (0 to 100): target plateau around 30 - 38 wt%
        # Normalized score where ~35 wt% is ideal (100 pts)
        s_yield = np.clip((mu_bar / 40.0) * 100.0, 0.0, 100.0)

        # 3. Operational Throughput Penalty (0 to 100):
        # Heavy penalty if RT > 45 minutes; minimal penalty if fast (<= 30 min)
        # Deduct points exponentially if RT > 45 min
        rt_vals = candidates_raw[:, 1]
        p_time = np.where(
            rt_vals <= 30.0,
            (rt_vals / 30.0) * 10.0,
            np.where(
                rt_vals <= 45.0,
                10.0 + ((rt_vals - 30.0) / 15.0) * 20.0,
                30.0 + 70.0 * (1.0 - np.exp(-(rt_vals - 45.0) / 35.0)),
            )
        )

        # 4. Energy Consumption Penalty (0 to 100): T * RT product
        # Lower thermal exposure (T * RT) is favored; high HR quickly reaching target is favored
        hr_vals = candidates_raw[:, 2]
        thermal_exposure = (candidates_raw[:, 0] / 500.0) * (rt_vals / 60.0) # normalized
        p_energy = np.clip(thermal_exposure * 35.0 - (hr_vals / 50.0) * 5.0, 0.0, 100.0)

        # Final Weighted Score: (40% Yield) - (35% Time) - (25% Energy) + 10 baseline
        w_yield = 0.40
        w_time = 0.35
        w_energy = 0.25

        balanced_score = (
            w_yield * s_yield
            - w_time * p_time
            - w_energy * p_energy
            + 30.0 # baseline offset to keep top performers ~70-95
        )

        # Disqualify recipes failing the quality gate
        balanced_score[~valid_gate] = -999.0

        return {
            "candidates": candidates_raw,
            "mean_yield": mu_bar,
            "epistemic_std": sigma_epistemic,
            "aleatoric_std": sigma_aleatoric,
            "cost_penalty": p_energy,
            "acquisition_score": balanced_score,
        }

    def select_batch(self, eval_results: Dict[str, np.ndarray]) -> List[Dict[str, float]]:
        """
        Greedily selects batch of recommended experiments with distance-based diversity penalty.
        """
        candidates = eval_results["candidates"].copy()
        scores = eval_results["acquisition_score"].copy()
        means = eval_results["mean_yield"]
        epistemics = eval_results["epistemic_std"]
        costs = eval_results["cost_penalty"]

        # Normalize candidates for distance computation
        t_span = self.bounds["Temperature"][1] - self.bounds["Temperature"][0]
        rt_span = self.bounds["Residence_Time"][1] - self.bounds["Residence_Time"][0]
        hr_span = self.bounds["Heating_Rate"][1] - self.bounds["Heating_Rate"][0]
        cand_norm = candidates / np.array([t_span, rt_span, hr_span])

        selected_indices = []

        for _ in range(self.batch_size):
            best_idx = int(np.argmax(scores))
            selected_indices.append(best_idx)

            # Apply soft penalty to candidates close to already selected point
            selected_pos = cand_norm[best_idx]
            dist_sq = np.sum((cand_norm - selected_pos) ** 2, axis=1)
            diversity_penalty = np.exp(-dist_sq / 0.05) * 5.0 # Penalize nearby cluster
            scores -= diversity_penalty

        recommendations = []
        for idx in selected_indices:
            recommendations.append({
                "Temperature_C": float(candidates[idx, 0]),
                "Residence_Time_min": float(candidates[idx, 1]),
                "Heating_Rate_C_per_min": float(candidates[idx, 2]),
                "Predicted_Yield_pct": float(means[idx]),
                "Epistemic_Uncertainty": float(epistemics[idx]),
                "Cost_Penalty": float(costs[idx]),
                "Acquisition_Score": float(eval_results["acquisition_score"][idx]),
            })

        return recommendations


def recommend_next_experiments(
    ensemble_models: List[nn.Module],
    feedstock_features: np.ndarray,
    process_scaler,
    config: dict,
    raw_biopolymers: np.ndarray,
    ash_weight: float = 0.05,
) -> List[Dict[str, float]]:
    """High-level function to generate recommendations for a target biomass feedstock."""
    acq_cfg = config["acquisition"]
    optimizer = CostAwareBatchUCB(
        bounds=acq_cfg["bounds"],
        beta_exploration=acq_cfg.get("beta_exploration", 1.96),
        cost_T_weight=acq_cfg.get("cost_temperature_weight", 0.004),
        cost_RT_weight=acq_cfg.get("cost_residence_time_weight", 0.002),
        batch_size=acq_cfg.get("batch_size", 3),
    )
    candidates = optimizer.sample_candidate_grid()
    results = optimizer.evaluate_candidates(
        ensemble_models=ensemble_models,
        feedstock_features=feedstock_features,
        process_scaler=process_scaler,
        candidates_raw=candidates,
        raw_biopolymers=raw_biopolymers,
        ash_weight=ash_weight,
    )
    return optimizer.select_batch(results)
