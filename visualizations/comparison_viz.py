"""
Comparison visualisations for the poster experiments.

Two functions:

1. plot_comparison_bars
   Bar chart comparing all policy×trainer combos across one or more metrics.
   One group of bars per metric; each bar = one (policy, trainer) combination.
   Error bars = std across seeds.

2. plot_lambda_sensitivity
   Dual-axis line plot showing how transport cost and demand satisfaction
   vary as lambda_penalty is swept over {0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0}.

Usage:
    from visualizations.comparison_viz import plot_comparison_bars, plot_lambda_sensitivity

    # results: list of dicts with keys: label, metric_name, mean, std
    fig = plot_comparison_bars(results, metrics=["episode_reward", "demand_satisfaction"])

    # lambda_results: list of dicts with keys: lambda, transport_cost, demand_satisfaction
    fig = plot_lambda_sensitivity(lambda_results)
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]

_METRIC_LABELS = {
    "episode_reward":       "Episode Reward",
    "total_transport_cost": "Transport Cost",
    "demand_satisfaction":  "Demand Satisfaction",
    "inventory_std":        "Inventory Std",
    "replenishment_cost":   "Replenishment Cost",
}


def plot_comparison_bars(
    results: list[dict],
    metrics: list[str] | None = None,
    save_path: str | None = None,
    wandb_log: bool = False,
    title: str | None = None,
) -> plt.Figure:
    """
    Grouped bar chart: policy×trainer combos on the x-axis, one subplot per metric.

    Each entry in `results` must have:
        label  : str   — e.g. "mlp_ppo"
        <metric>_mean : float
        <metric>_std  : float

    Parameters
    ----------
    results : list of result dicts
    metrics : metrics to plot (default: reward, cost, satisfaction)
    """
    if metrics is None:
        metrics = ["episode_reward", "total_transport_cost", "demand_satisfaction"]

    labels    = [r["label"] for r in results]
    n_agents  = len(labels)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4), squeeze=False)
    axes      = axes[0]
    x         = np.arange(n_agents)
    width     = 0.6

    for ax, metric in zip(axes, metrics):
        means = [r.get(f"{metric}_mean", r.get(metric, 0)) for r in results]
        stds  = [r.get(f"{metric}_std", 0) for r in results]
        colors = [_PALETTE[i % len(_PALETTE)] for i in range(n_agents)]

        bars = ax.bar(x, means, width, yerr=stds, color=colors,
                      capsize=4, edgecolor="white", linewidth=0.5)

        # Value labels on top of bars
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.05 + abs(mean) * 0.01,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=7.5
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(_METRIC_LABELS.get(metric, metric), fontsize=10)
        ax.set_title(_METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()

    _save_and_log(fig, save_path, wandb_log, "viz/comparison_bars")
    return fig


def plot_lambda_sensitivity(
    lambda_results: list[dict],
    save_path: str | None = None,
    wandb_log: bool = False,
    title: str | None = None,
) -> plt.Figure:
    """
    Dual-axis plot: transport cost (left) and demand satisfaction (right)
    vs. lambda_penalty on a log x-axis.

    Each entry in `lambda_results` must have:
        lambda                 : float
        transport_cost_mean    : float
        transport_cost_std     : float
        demand_satisfaction_mean : float
        demand_satisfaction_std  : float

    Example:
        lambda_results = [
            {"lambda": 0.01, "transport_cost_mean": 120.0, ... },
            {"lambda": 0.1,  ...},
            ...
        ]
    """
    lambdas  = [r["lambda"] for r in lambda_results]
    tc_mean  = np.array([r["transport_cost_mean"] for r in lambda_results])
    tc_std   = np.array([r.get("transport_cost_std", 0) for r in lambda_results])
    ds_mean  = np.array([r["demand_satisfaction_mean"] for r in lambda_results])
    ds_std   = np.array([r.get("demand_satisfaction_std", 0) for r in lambda_results])

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2      = ax1.twinx()

    color_tc = "#d62728"
    color_ds = "#1f77b4"

    ax1.semilogx(lambdas, tc_mean, "o-", color=color_tc, linewidth=2, label="Transport Cost")
    ax1.fill_between(lambdas, tc_mean - tc_std, tc_mean + tc_std, alpha=0.2, color=color_tc)
    ax1.set_xlabel("λ (lambda_penalty)", fontsize=11)
    ax1.set_ylabel("Transport Cost", fontsize=11, color=color_tc)
    ax1.tick_params(axis="y", labelcolor=color_tc)

    ax2.semilogx(lambdas, ds_mean, "s--", color=color_ds, linewidth=2, label="Demand Satisfaction")
    ax2.fill_between(lambdas, ds_mean - ds_std, ds_mean + ds_std, alpha=0.2, color=color_ds)
    ax2.set_ylabel("Demand Satisfaction Rate", fontsize=11, color=color_ds)
    ax2.tick_params(axis="y", labelcolor=color_ds)
    ax2.set_ylim(0, 1.05)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_title(title or "λ Sensitivity: Transport Cost vs. Demand Satisfaction", fontsize=11)
    fig.tight_layout()

    _save_and_log(fig, save_path, wandb_log, "viz/lambda_sensitivity")
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
