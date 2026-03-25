"""
Cost matrix visualisation — annotated heatmap with hub nodes highlighted.

Usage:
    from visualizations.cost_matrix_viz import plot_cost_matrix

    fig = plot_cost_matrix(env.cost_matrix, hub_fraction=0.2, save_path="outputs/viz/cost_matrix.png")
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import math
import os


def plot_cost_matrix(
    cost_matrix: np.ndarray,
    hub_fraction: float = 0.2,
    save_path: str | None = None,
    wandb_log: bool = False,
    title: str | None = None,
) -> plt.Figure:
    """
    Plot the N×N cost matrix as an annotated heatmap.

    Hub warehouses (first ceil(hub_fraction * N)) are highlighted with a
    coloured border so the hub-and-spoke structure is immediately visible.

    Parameters
    ----------
    cost_matrix  : (N, N) float array
    hub_fraction : fraction of warehouses that are hubs (matches env config)
    save_path    : if set, save figure to this path
    wandb_log    : if True, log figure to the active WandB run
    title        : optional plot title

    Returns
    -------
    matplotlib Figure
    """
    n      = cost_matrix.shape[0]
    n_hubs = max(1, math.ceil(hub_fraction * n))

    fig, ax = plt.subplots(figsize=(max(5, n), max(4, n - 1)))

    # Custom colormap: white (low cost) → deep red (high cost)
    cmap = LinearSegmentedColormap.from_list("cost", ["#ffffff", "#d73027"])
    im   = ax.imshow(cost_matrix, cmap=cmap, aspect="auto", vmin=0)

    # Annotate each cell with the cost value
    for i in range(n):
        for j in range(n):
            val   = cost_matrix[i, j]
            color = "white" if val > 0.7 * cost_matrix.max() else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    # Axis labels
    labels = [f"W{i}" + (" ★" if i < n_hubs else "") for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Destination", fontsize=10)
    ax.set_ylabel("Source", fontsize=10)

    # Draw thick border around hub rows/cols
    for idx in range(n_hubs):
        ax.add_patch(mpatches.Rectangle(
            (idx - 0.5, -0.5), 1, n,
            linewidth=2, edgecolor="#2166ac", facecolor="#2166ac22"
        ))
        ax.add_patch(mpatches.Rectangle(
            (-0.5, idx - 0.5), n, 1,
            linewidth=2, edgecolor="#2166ac", facecolor="#2166ac22"
        ))

    plt.colorbar(im, ax=ax, label="Shipping cost")

    hub_patch  = mpatches.Patch(color="#2166ac", alpha=0.4, label=f"Hub warehouses ({n_hubs})")
    ax.legend(handles=[hub_patch], loc="upper right", fontsize=8)

    ax.set_title(title or f"Shipping Cost Matrix (N={n}, {n_hubs} hubs)", fontsize=11)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if wandb_log:
        try:
            import wandb
            wandb.log({"viz/cost_matrix": wandb.Image(fig)})
        except Exception:
            pass

    return fig
