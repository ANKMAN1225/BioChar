"""
project/utils/transforms.py
Stateless compositional log-ratio transforms for biomass simplexes (ILR & CLR).
Supports NumPy arrays and PyTorch tensors.
"""

from typing import Union
import numpy as np
import torch


def _closure(x: Union[np.ndarray, torch.Tensor], eps: float = 1e-6) -> Union[np.ndarray, torch.Tensor]:
    """Ensures compositions sum to 1.0 with strictly positive values."""
    if isinstance(x, torch.Tensor):
        x_clipped = torch.clamp(x, min=eps)
        return x_clipped / torch.sum(x_clipped, dim=-1, keepdim=True)
    else:
        x_arr = np.asarray(x, dtype=np.float64)
        x_clipped = np.clip(x_arr, a_min=eps, a_max=None)
        return x_clipped / np.sum(x_clipped, axis=-1, keepdims=True)


def clr_transform(x: Union[np.ndarray, torch.Tensor], eps: float = 1e-6) -> Union[np.ndarray, torch.Tensor]:
    """
    Centered Log-Ratio (CLR) transformation.
    clr(x)_i = ln(x_i / g(x)), where g(x) is the geometric mean.
    Preserves dimension D, but lies on a hyperplane where sum(clr(x)) == 0.
    """
    if isinstance(x, torch.Tensor):
        closed = _closure(x, eps=eps)
        log_x = torch.log(closed)
        geom_mean_log = torch.mean(log_x, dim=-1, keepdim=True)
        return log_x - geom_mean_log
    else:
        closed = _closure(x, eps=eps)
        log_x = np.log(closed)
        geom_mean_log = np.mean(log_x, axis=-1, keepdims=True)
        return log_x - geom_mean_log


def inverse_clr(y: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Inverse CLR transform mapping back to the standard simplex."""
    if isinstance(y, torch.Tensor):
        exp_y = torch.exp(y)
        return exp_y / torch.sum(exp_y, dim=-1, keepdim=True)
    else:
        exp_y = np.exp(np.asarray(y, dtype=np.float64))
        return exp_y / np.sum(exp_y, axis=-1, keepdims=True)


def ilr_basis_matrix(D: int) -> np.ndarray:
    """
    Generates standard Helmert orthonormal basis matrix Psi of shape (D-1, D)
    satisfying Psi @ Psi.T = I_(D-1) and sum over rows = 0.
    """
    Psi = np.zeros((D - 1, D), dtype=np.float64)
    for i in range(1, D):
        r = i
        val_pos = 1.0 / np.sqrt(r * (r + 1))
        val_neg = -r / np.sqrt(r * (r + 1))
        Psi[i - 1, :r] = val_pos
        Psi[i - 1, r] = val_neg
    return Psi


def ilr_transform(x: Union[np.ndarray, torch.Tensor], eps: float = 1e-6) -> Union[np.ndarray, torch.Tensor]:
    """
    Isometric Log-Ratio (ILR) transformation.
    Maps D-part composition to an unconstrained (D-1)-dimensional Euclidean space.
    """
    if isinstance(x, torch.Tensor):
        D = x.shape[-1]
        psi = torch.tensor(ilr_basis_matrix(D), dtype=x.dtype, device=x.device)
        clr = clr_transform(x, eps=eps)
        return torch.matmul(clr, psi.T)
    else:
        x_arr = np.asarray(x, dtype=np.float64)
        D = x_arr.shape[-1]
        psi = ilr_basis_matrix(D)
        clr = clr_transform(x_arr, eps=eps)
        return np.dot(clr, psi.T)


def inverse_ilr(y: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Inverse ILR transform mapping (D-1) Euclidean coordinates back to the D-part simplex."""
    if isinstance(y, torch.Tensor):
        D = y.shape[-1] + 1
        psi = torch.tensor(ilr_basis_matrix(D), dtype=y.dtype, device=y.device)
        clr_reconstructed = torch.matmul(y, psi)
        return inverse_clr(clr_reconstructed)
    else:
        y_arr = np.asarray(y, dtype=np.float64)
        D = y_arr.shape[-1] + 1
        psi = ilr_basis_matrix(D)
        clr_reconstructed = np.dot(y_arr, psi)
        return inverse_clr(clr_reconstructed)
