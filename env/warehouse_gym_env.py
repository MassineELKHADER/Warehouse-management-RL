"""
Gymnasium wrapper around WarehouseEnv.

Converts the custom dict-based env into a standard gym.Env so it can be used
directly with Stable Baselines 3 (PPO, SAC, etc.) without any changes to the
core environment logic.

Observation space: Box(0, 1, shape=(2*N,)) — normalised [inventory, demand]
Action space:      Box(-inf, inf, shape=(N*N,)) — raw flat transport matrix;
                   WarehouseEnv._project_action() enforces all constraints.

Usage:
    from env.warehouse_gym_env import WarehouseGymEnv
    env = WarehouseGymEnv(cfg, seed=42)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import numpy as np
import gymnasium as gym

from env.warehouse_env import WarehouseEnv


class WarehouseGymEnv(gym.Env):
    """
    Gymnasium-compatible wrapper for WarehouseEnv.

    Parameters
    ----------
    cfg  : full config dict (same as passed to WarehouseEnv)
    seed : random seed
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, seed: int = 42):
        super().__init__()
        self._env  = WarehouseEnv(cfg, seed=seed)
        self._cfg  = cfg
        N           = self._env.n
        demand_mean = cfg["env"].get("demand_mean", 10.0)
        self._demand_scale = demand_mean * 3.0   # normalise demand to ~[0, 1]

        # Observation: [inventory/max_inv, demand/scale, step/episode_length]
        # The time feature lets the policy know urgency within an episode.
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(2 * N + 1,), dtype=np.float32
        )

        # Actions: [-1, 1] per edge. _project_action() clips negatives → 0
        # and rescales each row so total outgoing ≤ inventory.
        # Scale is arbitrary; the env normalises regardless.
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(N * N,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        state = self._env.reset()
        return self._obs(state), {}

    def step(self, action: np.ndarray):
        # De-normalise: action in [-1,1] → quantity in [0, max_inventory]
        # max(0, a) * max_inventory  so that:
        #   a ≤ 0  →  0 units shipped  (natural "do nothing" at init, mean≈0)
        #   a = 1  →  max_inventory units shipped
        # This avoids the previous bug where a=0 mapped to 250 units (half max).
        T_scaled = np.clip(action, 0.0, 1.0) * self._env.max_inventory
        T = T_scaled.reshape(self._env.n, self._env.n)
        next_state, reward, done, info = self._env.step(T)
        obs = self._obs(next_state)
        terminated = done
        truncated  = False
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass

    # ------------------------------------------------------------------
    # Accessors (useful for callbacks and visualisations)
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        return self._env.n

    @property
    def cost_matrix(self) -> np.ndarray:
        return self._env.cost_matrix

    @property
    def max_inventory(self) -> float:
        return self._env.max_inventory

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _obs(self, state: dict) -> np.ndarray:
        inv  = np.asarray(state["inventory"], dtype=np.float32) / self._env.max_inventory
        dem  = np.asarray(state["demand"],    dtype=np.float32) / self._demand_scale
        # Normalised time within episode: 0.0 at start → 1.0 at end
        t    = np.array([self._env._step_count / self._env.episode_length], dtype=np.float32)
        return np.concatenate([inv, dem, t])
