"""
Policy and value function visualisation.

Two main functions:

1. plot_policy_heatmap
   For MLP/AC policies: fix demand at its mean, sweep inventory[0] and inventory[1]
   over a grid, and show the total action magnitude (sum of T) as a heatmap.
   Tells you: "when warehouse 0 has X and warehouse 1 has Y, how much does
   the agent ship in total?"

   For GNN policies: plot the mean edge weight (shipping fraction) between
   every pair of warehouses as an N×N heatmap.

2. plot_value_surface
   For ActorCriticPolicy (has a value head): sweep a 2D (inventory, demand)
   grid and plot V(s) as a 3D surface or heatmap.
   Tells you: "what states does the agent consider most valuable?"

Usage:
    from visualizations.policy_viz import plot_policy_heatmap, plot_value_surface

    fig = plot_policy_heatmap(agent.policy, env, resolution=20)
    fig = plot_value_surface(agent.policy, env, resolution=30, mode="3d")
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch


def plot_policy_heatmap(
    policy,
    env,
    resolution: int = 20,
    save_path: str | None = None,
    wandb_log: bool = False,
    title: str | None = None,
) -> plt.Figure:
    """
    Visualise what action the policy takes across different inventory states.

    For flat-obs policies (MLP / AC):
        - Fix all inventories at max_inventory/2, fix all demands at demand_mean
        - Sweep inventory[0] and inventory[1] over [0, max_inventory]
        - Plot sum(T) (total shipping activity) as a heatmap

    For GNN policies:
        - Compute mean edge weights over a sample of random states
        - Plot as an N×N heatmap (similar to a cost matrix view)
    """
    from agents.policies.gnn_policy import GNNPolicy

    if isinstance(policy, GNNPolicy):
        return _gnn_edge_heatmap(policy, env, resolution, save_path, wandb_log, title)
    else:
        return _mlp_policy_heatmap(policy, env, resolution, save_path, wandb_log, title)


def _mlp_policy_heatmap(policy, env, resolution, save_path, wandb_log, title):
    max_inv    = env.max_inventory
    demand_mean = env._demand_model.mean if hasattr(env._demand_model, "mean") else 10.0

    inv_vals = np.linspace(0, max_inv, resolution)
    Z        = np.zeros((resolution, resolution))

    base_inventory = np.full(env.n, max_inv / 2, dtype=np.float32)
    base_demand    = np.full(env.n, demand_mean,  dtype=np.float32)

    for i, v0 in enumerate(inv_vals):
        for j, v1 in enumerate(inv_vals):
            inv           = base_inventory.copy()
            inv[0]        = v0
            if env.n > 1:
                inv[1] = v1
            state         = {"inventory": inv, "demand": base_demand}
            action, _, _  = policy.act(state, deterministic=True)
            T             = np.abs(action).reshape(env.n, env.n)
            Z[i, j]       = T.sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        Z, origin="lower", aspect="auto",
        extent=[0, max_inv, 0, max_inv],
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, label="Total shipped units")
    ax.set_xlabel("Inventory W1", fontsize=10)
    ax.set_ylabel("Inventory W0", fontsize=10)
    ax.set_title(title or "Policy: total shipping vs inventory", fontsize=11)
    fig.tight_layout()

    _save_and_log(fig, save_path, wandb_log, "viz/policy_heatmap")
    return fig


def _gnn_edge_heatmap(policy, env, resolution, save_path, wandb_log, title):
    """Average edge shipping weights over `resolution` random states."""
    n          = env.n
    edge_sums  = np.zeros((n, n), dtype=np.float64)

    rng = np.random.default_rng(0)
    for _ in range(resolution):
        inv   = rng.uniform(0, env.max_inventory, size=n).astype(np.float32)
        dem   = rng.uniform(0, 20, size=n).astype(np.float32)
        state = {"inventory": inv, "demand": dem}
        action, _, _ = policy.act(state, deterministic=True)
        T     = np.abs(action).reshape(n, n)
        np.fill_diagonal(T, 0)
        edge_sums += T

    edge_mean = edge_sums / resolution

    fig, ax = plt.subplots(figsize=(max(5, n), max(4, n - 1)))
    im = ax.imshow(edge_mean, cmap="Blues", aspect="auto")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{edge_mean[i,j]:.1f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="Mean shipped units")
    labels = [f"W{i}" for i in range(n)]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_xlabel("Destination"); ax.set_ylabel("Source")
    ax.set_title(title or "GNN policy: mean edge shipping", fontsize=11)
    fig.tight_layout()

    _save_and_log(fig, save_path, wandb_log, "viz/gnn_edge_heatmap")
    return fig


def plot_value_surface(
    policy,
    env,
    resolution: int = 30,
    mode: str = "heatmap",   # "heatmap" | "3d"
    save_path: str | None = None,
    wandb_log: bool = False,
    title: str | None = None,
) -> plt.Figure:
    """
    Visualise the value function V(inventory_0, demand_0) for ActorCriticPolicy.

    Sweeps inventory[0] and demand[0] over their natural ranges, holding all
    other warehouses fixed at their midpoints.

    Parameters
    ----------
    mode : "heatmap" for a 2D imshow, "3d" for a 3D surface plot
    """
    from agents.policies.actor_critic import ActorCriticPolicy

    if not isinstance(policy, ActorCriticPolicy):
        raise TypeError("plot_value_surface requires an ActorCriticPolicy with a value head.")

    max_inv    = env.max_inventory
    max_demand = 30.0
    inv_vals   = np.linspace(0, max_inv,    resolution)
    dem_vals   = np.linspace(0, max_demand, resolution)

    V = np.zeros((resolution, resolution), dtype=np.float32)
    base_obs = np.full(env.obs_dim, 0.5, dtype=np.float32)  # placeholder

    with torch.no_grad():
        for i, inv0 in enumerate(inv_vals):
            for j, dem0 in enumerate(dem_vals):
                obs         = base_obs.copy()
                obs[0]      = inv0 / max_inv   # normalise loosely
                obs[env.n]  = dem0 / max_demand
                obs_t       = torch.FloatTensor(obs).unsqueeze(0)
                V[i, j]     = policy.get_value(obs_t).item()

    if mode == "3d":
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        INV, DEM = np.meshgrid(inv_vals, dem_vals, indexing="ij")
        fig = plt.figure(figsize=(8, 6))
        ax  = fig.add_subplot(111, projection="3d")
        ax.plot_surface(INV, DEM, V, cmap="coolwarm", edgecolor="none", alpha=0.9)
        ax.set_xlabel("Inventory W0"); ax.set_ylabel("Demand W0"); ax.set_zlabel("V(s)")
        ax.set_title(title or "Value function surface V(inv₀, dem₀)", fontsize=11)
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            V, origin="lower", aspect="auto",
            extent=[0, max_demand, 0, max_inv],
            cmap="coolwarm",
        )
        plt.colorbar(im, ax=ax, label="V(s)")
        ax.set_xlabel("Demand W0"); ax.set_ylabel("Inventory W0")
        ax.set_title(title or "Value function heatmap V(inv₀, dem₀)", fontsize=11)

    fig.tight_layout()
    _save_and_log(fig, save_path, wandb_log, "viz/value_surface")
    return fig


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _save_and_log(fig: plt.Figure, save_path, wandb_log, wandb_key):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if wandb_log:
        try:
            import wandb
            wandb.log({wandb_key: wandb.Image(fig)})
        except Exception:
            pass
