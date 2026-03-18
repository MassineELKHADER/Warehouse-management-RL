"""
Comparison script — pulls all runs from WandB and generates plots.

Usage:
    python compare.py --scenario small
    python compare.py --scenario all

Produces:
  - Learning curves (episode_reward vs episode) per scenario
  - Bar chart: final demand_satisfaction per agent
  - Bar chart: final total_transport_cost per agent
  - Summary table printed to stdout
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

SCENARIOS = ["small", "medium", "large"]
METRICS = ["episode_reward", "total_transport_cost", "demand_satisfaction", "inventory_std"]


def fetch_runs(project: str, entity: str | None, scenario: str) -> pd.DataFrame:
    import wandb
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"tags": scenario})

    rows = []
    for run in runs:
        hist = run.history(keys=METRICS + ["episode"])
        hist["agent"] = run.config.get("agent", run.name)
        hist["scenario"] = run.config.get("scenario", scenario)
        hist["seed"] = run.config.get("seed", 0)
        rows.append(hist)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_learning_curves(df: pd.DataFrame, scenario: str, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for agent, grp in df.groupby("agent"):
        # Average across seeds
        curve = grp.groupby("episode")["episode_reward"].mean()
        ax.plot(curve.index, curve.values, label=agent)
    ax.set_title(f"Learning Curve — {scenario}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, f"learning_curve_{scenario}.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_final_bar(df: pd.DataFrame, metric: str, scenario: str, out_dir: str) -> None:
    # Use last 10% of episodes as "final"
    max_ep = df["episode"].max()
    cutoff = max_ep * 0.9
    final = df[df["episode"] >= cutoff].groupby("agent")[metric].mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    final.plot(kind="bar", ax=ax)
    ax.set_title(f"{metric} (final) — {scenario}")
    ax.set_ylabel(metric)
    ax.set_xlabel("Agent")
    fig.tight_layout()
    path = os.path.join(out_dir, f"bar_{metric}_{scenario}.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)


def print_summary_table(df: pd.DataFrame, scenario: str) -> None:
    max_ep = df["episode"].max()
    cutoff = max_ep * 0.9
    final = df[df["episode"] >= cutoff]
    table = final.groupby("agent")[METRICS].agg(["mean", "std"])
    print(f"\n=== Summary: {scenario} ===")
    print(table.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="all", choices=["all"] + SCENARIOS)
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    import wandb
    project = os.getenv("WANDB_PROJECT", "warehouse-rl")
    entity = os.getenv("WANDB_ENTITY", None)

    os.makedirs(args.out_dir, exist_ok=True)
    scenarios = SCENARIOS if args.scenario == "all" else [args.scenario]

    for sc in scenarios:
        print(f"\nFetching runs for scenario: {sc} ...")
        df = fetch_runs(project, entity, sc)
        if df.empty:
            print(f"  No runs found for scenario '{sc}'. Train some agents first.")
            continue

        plot_learning_curves(df, sc, args.out_dir)
        for metric in ["demand_satisfaction", "total_transport_cost"]:
            plot_final_bar(df, metric, sc, args.out_dir)
        print_summary_table(df, sc)


if __name__ == "__main__":
    main()
