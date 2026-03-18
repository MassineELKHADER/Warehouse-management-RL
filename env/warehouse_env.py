"""
Warehouse Redistribution Environment — P1 owns this file.

No gym dependency. Plain Python + NumPy.

State  : dict with keys
    "inventory" : (N,)  float32
    "demand"    : (N,)  float32

Action : (N, N) float32 transport matrix T
    T[i,j] = quantity shipped from warehouse i to warehouse j
    Constraint enforced inside step(): excess is clipped proportionally.

Reward : -sum_ij(c_ij * T_ij) - lambda * sum_i max(0, D_i - I_i_after_shipping)
"""

import numpy as np
from env.demand_models import make_demand_model


class WarehouseEnv:
    def __init__(self, cfg: dict, seed: int = 42):
        self.n = cfg["n_warehouses"]
        self.max_inventory = cfg.get("max_inventory", 100)
        self.lam = cfg.get("lambda_penalty", 2.0)
        self.episode_length = cfg.get("episode_length", 50)

        self._rng = np.random.default_rng(seed)
        self._demand_model = make_demand_model(cfg)

        # Cost matrix: symmetric, zeros on diagonal
        cost_override = cfg.get("cost_matrix", None)
        if cost_override is not None:
            self.cost_matrix = np.array(cost_override, dtype=np.float32)
        else:
            # Random symmetric costs in [0.5, 2.0], 0 on diagonal
            c = self._rng.uniform(0.5, 2.0, size=(self.n, self.n)).astype(np.float32)
            c = (c + c.T) / 2
            np.fill_diagonal(c, 0.0)
            self.cost_matrix = c

        self._inventory: np.ndarray = np.zeros(self.n, dtype=np.float32)
        self._demand: np.ndarray = np.zeros(self.n, dtype=np.float32)
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Reset to a fresh episode. Returns initial state dict."""
        if hasattr(self._demand_model, "reset"):
            self._demand_model.reset()

        self._inventory = self._rng.uniform(
            0, self.max_inventory, size=self.n
        ).astype(np.float32)
        self._demand = self._demand_model.sample(self.n, self._rng)
        self._step_count = 0
        return self._get_state()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """
        Apply transport matrix T (shape N×N), advance demand, return transition.

        Returns (next_state, reward, done, info)
        """
        T = self._project_action(action)

        # Shipping costs
        transport_cost = float(np.sum(self.cost_matrix * T))

        # Update inventory after outgoing/incoming shipments
        outgoing = T.sum(axis=1)   # (N,) shipped out from each warehouse
        incoming = T.sum(axis=0)   # (N,) received at each warehouse
        inventory_after_ship = self._inventory - outgoing + incoming

        # Satisfy demand
        satisfied = np.minimum(inventory_after_ship, self._demand)
        unmet_demand = np.maximum(self._demand - inventory_after_ship, 0.0)
        self._inventory = np.maximum(inventory_after_ship - satisfied, 0.0)

        # Sample next demand
        self._demand = self._demand_model.sample(self.n, self._rng)
        self._step_count += 1
        done = self._step_count >= self.episode_length

        reward = -transport_cost - self.lam * float(np.sum(unmet_demand))

        info = {
            "transport_cost": transport_cost,
            "unmet_demand": float(np.sum(unmet_demand)),
            "demand_satisfaction": float(np.sum(satisfied) / (np.sum(self._demand) + 1e-8)),
            "inventory": self._inventory.copy(),
            "action": T,
        }
        return self._get_state(), reward, done, info

    def sample_action(self) -> np.ndarray:
        """Return a random valid transport matrix."""
        T = np.zeros((self.n, self.n), dtype=np.float32)
        for i in range(self.n):
            if self._inventory[i] > 0:
                # Random fractions summing to at most 1
                fracs = self._rng.dirichlet(np.ones(self.n)) * self._rng.uniform(0, 1)
                fracs[i] = 0.0  # no self-shipment
                T[i] = fracs * self._inventory[i]
        return T

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict:
        return self._get_state()

    @property
    def action_shape(self) -> tuple:
        return (self.n, self.n)

    @property
    def obs_dim(self) -> int:
        """Flattened observation dimension (2*N for inventory + demand)."""
        return 2 * self.n

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_state(self) -> dict:
        return {
            "inventory": self._inventory.copy(),
            "demand": self._demand.copy(),
        }

    def _project_action(self, action: np.ndarray) -> np.ndarray:
        """
        Clip/scale action so that:
          - T[i,j] >= 0
          - T[i,i] = 0  (no self-shipment)
          - sum_j T[i,j] <= inventory[i]  (can't ship more than available)
        """
        T = np.array(action, dtype=np.float32)
        T = np.clip(T, 0.0, None)
        np.fill_diagonal(T, 0.0)

        row_sums = T.sum(axis=1)
        for i in range(self.n):
            if row_sums[i] > self._inventory[i]:
                T[i] *= self._inventory[i] / (row_sums[i] + 1e-8)
        return T

    def flat_obs(self, state: dict) -> np.ndarray:
        """Convert state dict to flat numpy array for MLP-based agents."""
        return np.concatenate([state["inventory"], state["demand"]])
