"""
Quick visualisation CLI.

Usage:
    python visualize.py cost_matrix                         # small config, seed 42
    python visualize.py cost_matrix --config medium --seed 7
    python visualize.py cost_matrix --config large --save   # save to outputs/viz/

    python visualize.py shipping_graph --config small       # random policy snapshot
    python visualize.py shipping_graph --config small --checkpoint outputs/sb3_ppo_small_s42.zip --algo ppo
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


def cmd_shipping_graph(args):
    import matplotlib.pyplot as plt
    import numpy as np
    from env.warehouse_gym_env import WarehouseGymEnv
    from visualizations.shipping_graph_viz import plot_shipping_graph

    cfg = load_config(args.config)
    env = WarehouseGymEnv(cfg, seed=args.seed)
    obs, _ = env.reset()

    # Load model or use random actions
    if args.checkpoint:
        algo = (args.algo or "ppo").lower()
        if algo == "ppo":
            from stable_baselines3 import PPO
            model = PPO.load(args.checkpoint, env=env)
        elif algo == "sac":
            from stable_baselines3 import SAC
            model = SAC.load(args.checkpoint, env=env)
        else:
            raise ValueError(f"Unknown --algo: {algo}")
        def predict(o): return model.predict(o, deterministic=True)[0]
    else:
        print("No --checkpoint given, using random actions.")
        def predict(_): return env.action_space.sample()

    # Run one episode, collect the average T matrix
    T_sum  = np.zeros((env.n, env.n), dtype=np.float64)
    steps  = 0
    done   = False
    last_inv = None
    while not done:
        action = predict(obs)
        obs, _, terminated, truncated, info = env.step(action)
        T_sum += np.abs(action.reshape(env.n, env.n))
        steps += 1
        done   = terminated or truncated
        last_inv = info["inventory"]

    T_avg = T_sum / max(steps, 1)

    save_path = None
    if args.save:
        tag = "random" if not args.checkpoint else (args.algo or "ppo")
        save_path = f"outputs/viz/shipping_graph_{args.config}_{tag}_s{args.seed}.png"

    fig = plot_shipping_graph(
        T_avg,
        env.cost_matrix,
        hub_fraction=cfg.get("hub_fraction", 0.2),
        inventory=last_inv,
        title=f"Avg shipping policy — {args.config} (N={env.n})",
        save_path=save_path,
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

    p_sg = sub.add_parser("shipping_graph", help="Plot policy as a weighted directed graph")
    p_sg.add_argument("--config",     default="small", choices=["default", "small", "medium", "large"])
    p_sg.add_argument("--seed",       type=int, default=42)
    p_sg.add_argument("--checkpoint", default=None, help="Path to SB3 .zip checkpoint")
    p_sg.add_argument("--algo",       default="ppo", choices=["ppo", "sac"],
                      help="SB3 algorithm used (needed to load checkpoint)")
    p_sg.add_argument("--save",       action="store_true", help="Save figure to outputs/viz/")

    args = parser.parse_args()

    if args.command == "cost_matrix":
        cmd_cost_matrix(args)
    elif args.command == "shipping_graph":
        cmd_shipping_graph(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
