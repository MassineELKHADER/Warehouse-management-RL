"""
Shipping graph visualisation — directed weighted graph of the transport matrix.

Nodes = warehouses.  Edges = T_{i,j} (units shipped from i to j).

Visual encoding:
    Node size     : proportional to current inventory level
    Node colour   : green  if inventory >= demand (no shortage)
                    red    if inventory <  demand (shortage)
    Hub nodes     : drawn with a star marker and bold border
    Edge thickness: proportional to T_{i,j}
    Edge colour   : blue (cheap route) → red (expensive route)
    Edge label    : T_{i,j} value (shown only if T_{i,j} > threshold)

Works for any policy (MLP+PPO, GNN+REINFORCE, SB3, etc.) — just pass
the T matrix from one evaluation step or episode average.

Usage:
    from visualizations.shipping_graph_viz import plot_shipping_graph

    fig = plot_shipping_graph(T, env.cost_matrix,
                              inventory=state["inventory"],
                              demand=state["demand"])
    wandb.log({"eval/shipping_graph": wandb.Image(fig)})
"""

import os
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def plot_shipping_graph(
    T: np.ndarray,
    cost_matrix: np.ndarray,
    hub_fraction: float = 0.2,
    inventory: np.ndarray | None = None,
    demand: np.ndarray | None = None,
    edge_threshold: float = 0.01,   # edges below this fraction of max(T) are hidden
    save_path: str | None = None,
    wandb_log: bool = False,
    title: str | None = None,
) -> plt.Figure:
    """
    Draw the transport matrix T as a directed weighted graph.

    Parameters
    ----------
    T             : (N, N) transport matrix (after _project_action)
    cost_matrix   : (N, N) shipping costs (used to colour edges)
    hub_fraction  : fraction of warehouses that are hubs (≈ 0.2)
    inventory     : (N,) current inventory per warehouse (sizes nodes)
    demand        : (N,) current demand per warehouse (colours nodes)
    edge_threshold: hide edges smaller than this fraction of max(T)
    save_path     : if set, save to this path
    wandb_log     : if True, log to active WandB run
    title         : optional plot title
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required: pip install networkx")

    N      = T.shape[0]
    n_hubs = max(1, math.ceil(hub_fraction * N))
    T_max  = T.max() if T.max() > 0 else 1.0
    thresh = edge_threshold * T_max

    # --- Build directed graph -------------------------------------------
    G = nx.DiGraph()
    G.add_nodes_from(range(N))

    for i in range(N):
        for j in range(N):
            if i != j and T[i, j] > thresh:
                G.add_edge(i, j, weight=T[i, j], cost=cost_matrix[i, j])

    # --- Layout ----------------------------------------------------------
    # Circular layout, hubs placed at inner ring
    if N <= 10:
        pos = nx.circular_layout(G)
    else:
        pos = nx.shell_layout(
            G,
            nlist=[list(range(n_hubs)), list(range(n_hubs, N))]
        )

    # --- Node attributes -------------------------------------------------
    if inventory is not None:
        max_inv    = inventory.max() if inventory.max() > 0 else 1.0
        node_sizes = (inventory / max_inv * 1200 + 300).tolist()
    else:
        node_sizes = [600] * N

    if demand is not None and inventory is not None:
        # Red = shortage, green = surplus
        node_colors = [
            "#d62728" if inventory[i] < demand[i] else "#2ca02c"
            for i in range(N)
        ]
    else:
        node_colors = ["#1f77b4"] * N

    # Hubs get a distinct colour override (gold)
    for i in range(n_hubs):
        node_colors[i] = "#ff7f0e"

    # --- Edge attributes -------------------------------------------------
    edge_widths = []
    edge_colors = []
    cost_norm   = mcolors.Normalize(
        vmin=cost_matrix.min(), vmax=cost_matrix.max()
    )
    cmap_edge = cm.get_cmap("RdYlBu_r")   # blue=cheap, red=expensive

    for u, v, data in G.edges(data=True):
        w = data["weight"]
        edge_widths.append(1.0 + 5.0 * (w / T_max))
        edge_colors.append(cmap_edge(cost_norm(data["cost"])))

    # --- Draw ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(6, N), max(5, N - 1)))

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="black", linewidths=1.2,
    )

    # Hub nodes with star shape
    hub_pos = {i: pos[i] for i in range(n_hubs)}
    nx.draw_networkx_nodes(
        G, hub_pos, ax=ax,
        nodelist=list(range(n_hubs)),
        node_size=[node_sizes[i] * 1.3 for i in range(n_hubs)],
        node_color=[node_colors[i] for i in range(n_hubs)],
        node_shape="*",
        edgecolors="black", linewidths=1.5,
    )

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=15,
        connectionstyle="arc3,rad=0.15",
    )

    # Node labels: "W0 (hub)" for hubs
    labels = {
        i: f"W{i}{'★' if i < n_hubs else ''}"
        for i in range(N)
    }
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8, font_weight="bold")

    # Edge labels: show T value
    edge_labels = {
        (u, v): f"{data['weight']:.1f}"
        for u, v, data in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels, ax=ax, font_size=7, alpha=0.8
    )

    # --- Legend ----------------------------------------------------------
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#ff7f0e", label="Hub warehouse"),
        Patch(facecolor="#2ca02c", label="No shortage"),
        Patch(facecolor="#d62728", label="Shortage (inv < demand)"),
    ]
    ax.legend(handles=legend_els, loc="upper left", fontsize=8)

    # Colourbar for edge cost
    sm = cm.ScalarMappable(cmap=cmap_edge, norm=cost_norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Shipping cost", shrink=0.6, pad=0.02)

    ax.set_title(title or f"Shipping policy graph (N={N})", fontsize=11)
    ax.axis("off")
    fig.tight_layout()

    # --- Save / log ------------------------------------------------------
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if wandb_log:
        try:
            import wandb
            wandb.log({"viz/shipping_graph": wandb.Image(fig)})
        except Exception:
            pass

    return fig
