"""
Greedy baseline: each warehouse ships its surplus inventory
(inventory - own demand) to warehouses with deficits,
weighted by deficit / shipping_cost.
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

        surplus = np.maximum(inventory - demand, 0.0)   # what each warehouse can afford to send
        deficit = np.maximum(demand - inventory, 0.0)   # what each warehouse still needs

        for i in range(n):
            if surplus[i] <= 0:
                continue

            # Score destinations by deficit / cost — high deficit, low cost = ship there
            scores = deficit.copy().astype(np.float32)
            scores[i] = 0.0  # don't ship to self
            costs_i = self.cost_matrix[i]
            with np.errstate(divide="ignore", invalid="ignore"):
                scores = np.where(costs_i > 0, scores / (costs_i + 1e-8), 0.0)

            if scores.sum() == 0:
                continue

            weights = scores / scores.sum()
            T[i] = weights * surplus[i]

        return T

    def update(self, batch: dict) -> dict:
        return {}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

if __name__ == "__main__":
    import numpy as np
    cost_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    agent = GreedyAgent(cost_matrix=cost_matrix)
    state = {
        "inventory": np.array([60.0, 10.0], dtype=np.float32),
        "demand": np.array([50.0, 20.0], dtype=np.float32),
    }
    action = agent.act(state)
    print("Action (transport matrix):")
    print(action)
    