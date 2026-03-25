"""
Training visualisation — learning curves with mean ± std shading across seeds.

Usage:
    from visualizations.training_viz import plot_learning_curves

    # runs: list of dicts, each with keys "episodes" and metric arrays
    runs = [
        {"label": "mlp_ppo", "episodes": [...], "episode_reward": [...], ...},
        ...
    ]
    fig = plot_learning_curves(runs, metric="episode_reward")
"""

import os
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


_METRICS_LABELS = {
    "episode_reward":      "Episode Reward",
    "total_transport_cost": "Total Transport Cost",
    "demand_satisfaction":  "Demand Satisfaction Rate",
    "inventory_std":        "Inventory Std (balance)",
    "replenishment_cost":   "External Replenishment Cost",
    "entropy":              "Policy Entropy",
    "policy_loss":          "Policy Loss",
    "value_loss":           "Value Loss",
}

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def plot_learning_curves(
    runs: list[dict],
    metrics: str | list[str] = "episode_reward",
    smooth_window: int = 10,
    save_path: str | None = None,
    wandb_log: bool = False,
    title: str | None = None,
) -> plt.Figure:
    """
    Plot one or several learning curve metrics for a list of runs.

    Each entry in `runs` must have:
        label     : str  — display name (e.g. "mlp_ppo")
        episodes  : list[int] — x-axis
        <metric>  : list[float] — y-axis values (one per episode checkpoint)

    For runs with multiple seeds, pass one entry per seed sharing the same
    label; the function groups by label and plots mean ± std shading.

    Parameters
    ----------
    runs          : list of run dicts (see above)
    metrics       : single metric name or list of metrics to plot (subplots)
    smooth_window : rolling average window (set to 1 to disable)
    save_path     : if set, save to this path
    wandb_log     : log figure to active WandB run
    """
    if isinstance(metrics, str):
        metrics = [metrics]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 4), squeeze=False)
    axes      = axes[0]

    # Group runs by label for multi-seed averaging
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        groups[run["label"]].append(run)

    for ax, metric in zip(axes, metrics):
        for ci, (label, group_runs) in enumerate(groups.items()):
            color = _PALETTE[ci % len(_PALETTE)]

            # Align all seeds to common episode checkpoints (use first seed's)
            all_y = []
            x_ref = group_runs[0]["episodes"]
            for run in group_runs:
                y = np.array(run.get(metric, []))
                if smooth_window > 1 and len(y) >= smooth_window:
                    y = np.convolve(y, np.ones(smooth_window) / smooth_window, mode="valid")
                    x = run["episodes"][smooth_window - 1:]
                else:
                    x = run["episodes"]
                # Interpolate to common x grid
                all_y.append(np.interp(x_ref, x, y))

            all_y  = np.array(all_y)   # (n_seeds, T)
            mean_y = all_y.mean(axis=0)
            std_y  = all_y.std(axis=0)

            ax.plot(x_ref, mean_y, label=label, color=color, linewidth=1.8)
            ax.fill_between(x_ref, mean_y - std_y, mean_y + std_y,
                            alpha=0.2, color=color)

        ax.set_xlabel("Episode", fontsize=10)
        ax.set_ylabel(_METRICS_LABELS.get(metric, metric), fontsize=10)
        ax.set_title(_METRICS_LABELS.get(metric, metric), fontsize=11)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if wandb_log:
        try:
            import wandb
            key = "viz/learning_curves_" + "_".join(metrics)
            wandb.log({key: wandb.Image(fig)})
        except Exception:
            pass

    return fig
