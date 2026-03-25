"""
Quick visualisation CLI.

Usage:
    python visualize.py cost_matrix                         # small config, seed 42
    python visualize.py cost_matrix --config medium --seed 7
    python visualize.py cost_matrix --config large --save   # save to outputs/viz/
"""

import argparse
import os
import yaml


def load_config(name: str) -> dict:
    with open(os.path.join("configs", "default.yaml")) as f:
        cfg = yaml.safe_load(f)
    if name != "default":
        with open(os.path.join("configs", f"{name}.yaml")) as f:
            override = yaml.safe_load(f)
        for key, val in override.items():
            if key == "_base_":
                continue
            if isinstance(val, dict) and key in cfg:
                cfg[key].update(val)
            else:
                cfg[key] = val
    return cfg


def cmd_cost_matrix(args):
    import matplotlib.pyplot as plt
    from env.warehouse_env import WarehouseEnv
    from visualizations.cost_matrix_viz import plot_cost_matrix

    cfg = load_config(args.config)
    env = WarehouseEnv(cfg, seed=args.seed)

    save_path = None
    if args.save:
        save_path = f"outputs/viz/cost_matrix_{args.config}_s{args.seed}.png"

    fig = plot_cost_matrix(
        env.cost_matrix,
        hub_fraction=cfg.get("hub_fraction", 0.2),
        save_path=save_path,
        title=f"Cost matrix - {args.config} (N={env.n})",
    )

    if save_path:
        print(f"Saved: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    p_cm = sub.add_parser("cost_matrix", help="Plot the N×N shipping cost heatmap")
    p_cm.add_argument("--config", default="small", choices=["default", "small", "medium", "large"])
    p_cm.add_argument("--seed",   type=int, default=42)
    p_cm.add_argument("--save",   action="store_true", help="Save figure to outputs/viz/")

    args = parser.parse_args()

    if args.command == "cost_matrix":
        cmd_cost_matrix(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
