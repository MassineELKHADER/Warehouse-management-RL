import numpy as np
from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Ships random valid quantities between warehouses."""

    def act(self, state: dict) -> np.ndarray:
        n = len(state["inventory"])
        inventory = state["inventory"]
        T = np.zeros((n, n), dtype=np.float32)
        rng = np.random.default_rng()
        for i in range(n):
            if inventory[i] > 0:
                fracs = rng.dirichlet(np.ones(n)) * rng.uniform(0, 1)
                fracs[i] = 0.0
                T[i] = fracs * inventory[i]
        return T

    def update(self, batch: dict) -> dict:
        return {}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
