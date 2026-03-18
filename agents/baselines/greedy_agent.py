"""
Greedy baseline: for each warehouse i with surplus inventory,
ship as much as possible to the warehouse with the highest unmet demand,
weighted inversely by shipping cost.
"""

import numpy as np
from agents.base_agent import BaseAgent


class GreedyAgent(BaseAgent):
    def __init__(self, cost_matrix: np.ndarray):
        self.cost_matrix = cost_matrix  # (N, N)

    def act(self, state: dict) -> np.ndarray:
        inventory = state["inventory"]  # (N,)
        demand = state["demand"]        # (N,)
        n = len(inventory)
        T = np.zeros((n, n), dtype=np.float32)

        # Score: demand / (cost + eps) — higher is better destination
        for i in range(n):
            available = inventory[i]
            if available <= 0:
                continue

            scores = demand.copy().astype(np.float32)
            scores[i] = 0.0  # don't ship to self
            costs_i = self.cost_matrix[i]
            # Avoid dividing by zero for same-warehouse (already 0 score)
            with np.errstate(divide="ignore", invalid="ignore"):
                scores = np.where(costs_i > 0, scores / (costs_i + 1e-8), 0.0)

            if scores.sum() == 0:
                continue

            # Distribute proportionally to score
            weights = scores / scores.sum()
            T[i] = weights * available

        return T

    def update(self, batch: dict) -> dict:
        return {}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
