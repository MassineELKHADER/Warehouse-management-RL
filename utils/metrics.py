import numpy as np


def total_transport_cost(actions: np.ndarray, cost_matrix: np.ndarray) -> float:
    """
    Sum of c_ij * T_ij over all (i,j) pairs for a single step.

    actions     : (N, N) transport matrix
    cost_matrix : (N, N) shipping cost matrix
    """
    return float(np.sum(cost_matrix * actions))


def demand_satisfaction_rate(inventory_after: np.ndarray, demand: np.ndarray) -> float:
    """
    Fraction of total demand that was met.

    inventory_after : (N,) inventory after shipping, before demand is subtracted
    demand          : (N,) demand at this step
    """
    satisfied = np.sum(np.minimum(inventory_after, demand))
    total_demand = np.sum(demand)
    if total_demand == 0:
        return 1.0
    return float(satisfied / total_demand)


def inventory_balance(inventory: np.ndarray) -> float:
    """Std of inventory across warehouses — lower means more balanced."""
    return float(np.std(inventory))


def episode_summary(
    rewards: list,
    costs: list,
    satisfaction_rates: list,
    inventories: list,
) -> dict:
    """Aggregate per-step lists into a single episode metrics dict."""
    return {
        "episode_reward": float(np.sum(rewards)),
        "total_transport_cost": float(np.sum(costs)),
        "demand_satisfaction": float(np.mean(satisfaction_rates)),
        "inventory_std": float(np.mean([inventory_balance(inv) for inv in inventories])),
    }
