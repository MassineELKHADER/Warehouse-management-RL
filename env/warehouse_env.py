"""
WarehouseEnv — core environment for multi-warehouse inventory redistribution.

State  : dict with keys
    "inventory" : (N,)  float32  — current stock at each warehouse
    "demand"    : (N,)  float32  — demand forecast for this step

Action : (N, N) float32 transport matrix T
    T[i,j] = quantity shipped from warehouse i to warehouse j
    Constraints are enforced inside step() via _project_action() — agents
    do NOT need to enforce them themselves.

Reward :
    -sum_ij(c_ij * T_ij)                             transport cost
    - lambda * sum_i max(0, D_i - I_i_post_ship)     unmet demand penalty
    - ext_cost * sum_i replenishment_i               [optional] external supplier

External supplier:
    If env.external_supplier.enabled is true, any warehouse whose inventory
    falls short of demand after shipping is automatically restocked at
    cost_per_unit >> max(c_ij). This prevents permanent inventory depletion
    over long episodes. Agents learn to avoid it because the penalty is large.

NOTE: this class expects the FULL config dict (not cfg["env"]).
"""

import numpy as np
from env.demand_models import make_demand_model
from env.cost_matrix import make_cost_matrix


class WarehouseEnv:
    def __init__(self, cfg: dict, seed: int = 42):
        env_cfg = cfg["env"]

        self.n               = env_cfg["n_warehouses"]
        self.max_inventory   = env_cfg.get("max_inventory", 100)
        self.lam             = env_cfg.get("lambda_penalty", 2.0)
        self.episode_length  = env_cfg.get("episode_length", 50)

        # --- External supplier (optional) --------------------------------
        ext_cfg = env_cfg.get("external_supplier", {})
        self._ext_enabled = ext_cfg.get("enabled", False)
        self._ext_cost    = float(ext_cfg.get("cost_per_unit", 50.0))

        # --- Reproducible randomness -------------------------------------
        self._rng = np.random.default_rng(seed)

        # --- Demand model and cost matrix --------------------------------
        self._demand_model = make_demand_model(cfg)
        self.cost_matrix   = make_cost_matrix(cfg, self._rng)

        # --- State variables ---------------------------------------------
        self._inventory:  np.ndarray = np.zeros(self.n, dtype=np.float32)
        self._demand:     np.ndarray = np.zeros(self.n, dtype=np.float32)
        self._step_count: int        = 0

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
        self._demand    = self._demand_model.sample(self.n, self._rng)
        self._step_count = 0
        return self._get_state()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """
        Apply transport matrix T (shape N×N), advance demand, return transition.

        Returns (next_state, reward, done, info).
        """
        T = self._project_action(action)

        # Shipping costs
        transport_cost = float(np.sum(self.cost_matrix * T))

        # Inventory update after outgoing / incoming shipments
        outgoing           = T.sum(axis=1)
        incoming           = T.sum(axis=0)
        inventory_post_ship = self._inventory - outgoing + incoming

        # Satisfy demand (using current-step demand, before sampling next)
        current_demand = self._demand.copy()
        satisfied      = np.minimum(inventory_post_ship, current_demand)
        unmet          = np.maximum(current_demand - inventory_post_ship, 0.0)
        self._inventory = np.maximum(inventory_post_ship - satisfied, 0.0)

        # External supplier restocks shortfall at high cost (optional)
        replenishment_cost = 0.0
        if self._ext_enabled and unmet.sum() > 0:
            self._inventory   += unmet          # bring each warehouse to 0 shortfall
            replenishment_cost = self._ext_cost * float(unmet.sum())

        # Advance to next step
        self._demand     = self._demand_model.sample(self.n, self._rng)
        self._step_count += 1
        done = self._step_count >= self.episode_length

        reward = (
            -transport_cost
            - self.lam * float(np.sum(unmet))
            - replenishment_cost
        )

        info = {
            "transport_cost":      transport_cost,
            "unmet_demand":        float(np.sum(unmet)),
            "demand_satisfaction": float(
                np.sum(satisfied) / (float(np.sum(current_demand)) + 1e-8)
            ),
            "inventory":           self._inventory.copy(),
            "action":              T,
            "replenishment_cost":  replenishment_cost,
        }
        return self._get_state(), reward, done, info

    def sample_action(self) -> np.ndarray:
        """Return a random valid transport matrix."""
        T = np.zeros((self.n, self.n), dtype=np.float32)
        for i in range(self.n):
            if self._inventory[i] > 0:
                fracs    = self._rng.dirichlet(np.ones(self.n)) * self._rng.uniform(0, 1)
                fracs[i] = 0.0  # no self-shipment
                T[i]     = fracs * self._inventory[i]
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
        """Flattened observation dimension: inventory + demand = 2*N."""
        return 2 * self.n

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_state(self) -> dict:
        return {
            "inventory": self._inventory.copy(),
            "demand":    self._demand.copy(),
        }

    def _project_action(self, action: np.ndarray) -> np.ndarray:
        """
        Project an arbitrary action into the feasible set:
          T[i,j] >= 0             no negative shipments
          T[i,i]  = 0             no self-shipment
          sum_j T[i,j] <= I[i]   can't ship more than available

        This is the single source of constraint enforcement.
        Agents return raw unconstrained values; the env cleans them here.
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
        """Convert state dict to flat (2*N,) array for MLP-based agents."""
        return np.concatenate([state["inventory"], state["demand"]])
