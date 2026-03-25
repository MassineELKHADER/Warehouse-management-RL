"""
Cost matrix factory for WarehouseEnv.

Three modes (set via `cost_matrix` key in config):
  - null / "random" : symmetric random matrix, Uniform(0.5, 2.0)
  - "hub"           : hub-and-spoke structure — ceil(hub_fraction * N) hubs
                      with tiered costs (hub↔hub cheap, spoke↔spoke expensive)
  - list[list]      : explicit N×N matrix provided directly in config

Usage:
    from env.cost_matrix import make_cost_matrix, get_hub_indices
    C = make_cost_matrix(cfg, rng)
"""

import math
import numpy as np


def make_cost_matrix(cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """
    Build and return the (N, N) float32 cost matrix.

    Reads:
      cfg["env"]["n_warehouses"]       — number of warehouses
      cfg.get("cost_matrix", None)     — mode: None | "random" | "hub" | list[list]
      cfg.get("hub_fraction", 0.2)     — fraction of warehouses that are hubs (hub mode only)
    """
    n = cfg["env"]["n_warehouses"]
    mode = cfg.get("cost_matrix", None)

    if mode is None or mode == "random":
        return _random_cost_matrix(n, rng)

    if mode == "hub":
        hub_fraction = float(cfg.get("hub_fraction", 0.2))
        return _hub_cost_matrix(n, rng, hub_fraction)

    # Explicit N×N list provided in config
    C = np.array(mode, dtype=np.float32)
    if C.shape != (n, n):
        raise ValueError(f"cost_matrix shape {C.shape} does not match n_warehouses={n}")
    return C


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _random_cost_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    """Symmetric random cost matrix, Uniform(0.5, 2.0), zero diagonal."""
    c = rng.uniform(0.5, 2.0, size=(n, n)).astype(np.float32)
    c = (c + c.T) / 2.0
    np.fill_diagonal(c, 0.0)
    return c


def _hub_cost_matrix(n: int, rng: np.random.Generator, hub_fraction: float = 0.2) -> np.ndarray:
    """
    Hub-and-spoke cost matrix.

    The first ceil(hub_fraction * N) warehouses are hubs.
    Cost tiers (symmetric):
      hub  ↔ hub  : Uniform(0.1, 0.3)  — cheapest
      hub  ↔ spoke: Uniform(0.3, 0.8)  — medium
      spoke↔ spoke: Uniform(1.0, 2.0)  — most expensive

    Args:
        n            : number of warehouses
        rng          : numpy Generator for reproducibility
        hub_fraction : fraction of warehouses that are hubs (default 0.2 → ~20%)

    Hub counts per scenario:
        small  (N=4)  → 1 hub
        medium (N=9)  → 2 hubs
        large  (N=16) → 4 hubs
    """
    n_hubs = max(1, math.ceil(hub_fraction * n))
    hub_set = set(range(n_hubs))

    c = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            i_hub = i in hub_set
            j_hub = j in hub_set
            if i_hub and j_hub:
                cost = float(rng.uniform(0.1, 0.3))
            elif i_hub or j_hub:
                cost = float(rng.uniform(0.3, 0.8))
            else:
                cost = float(rng.uniform(1.0, 2.0))
            c[i, j] = c[j, i] = cost
    return c


def get_hub_indices(n: int, hub_fraction: float = 0.2) -> list[int]:
    """Return the indices of hub warehouses for a given N.

    Useful for visualisations (highlight hub nodes) and for constructing
    obs_extractor variants that include hub membership as a feature.
    """
    return list(range(max(1, math.ceil(hub_fraction * n))))
