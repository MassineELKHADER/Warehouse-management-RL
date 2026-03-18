"""
Action-space utilities for model-free RL agents (P2).

The transport matrix T must satisfy:
    T[i,j] >= 0
    T[i,i] = 0
    sum_j T[i,j] <= inventory[i]

Two helpers:
  - project_action : hard clipping (used at inference)
  - flat_to_matrix / matrix_to_flat : convert between (N*N,) vectors
    (for MLP policy output) and (N,N) matrices
"""

import numpy as np
import torch


def project_action(T: np.ndarray, inventory: np.ndarray) -> np.ndarray:
    """Clip/scale T so it satisfies all constraints."""
    T = np.clip(T, 0.0, None)
    np.fill_diagonal(T, 0.0)
    row_sums = T.sum(axis=1)
    for i in range(len(inventory)):
        if row_sums[i] > inventory[i]:
            T[i] *= inventory[i] / (row_sums[i] + 1e-8)
    return T.astype(np.float32)


def flat_to_matrix(flat: np.ndarray, n: int) -> np.ndarray:
    """Reshape flat (N*N,) vector → (N,N) matrix."""
    return flat.reshape(n, n)


def matrix_to_flat(T: np.ndarray) -> np.ndarray:
    """Reshape (N,N) matrix → flat (N*N,) vector."""
    return T.flatten()


def mask_diagonal(T: torch.Tensor) -> torch.Tensor:
    """Zero the diagonal of a (B, N, N) or (N, N) tensor in-place."""
    n = T.shape[-1]
    idx = torch.arange(n, device=T.device)
    T[..., idx, idx] = 0.0
    return T
