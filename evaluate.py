"""
Shared evaluation script — runs N deterministic episodes, prints + logs metrics.

Usage:
    python evaluate.py --agent ppo --config small --seed 42 --checkpoint outputs/ppo_small_42.pt
    python evaluate.py --agent random --config small --seed 42  # no checkpoint needed
"""

import argparse
import os
import yaml
import numpy as np
from dotenv import load_dotenv

load_dotenv()


def load_config(config_name: str) -> dict:
    base_path = os.path.join("configs", "default.yaml")
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    if config_name != "default":
        cfg_path = os.path.join("configs", f"{config_name}.yaml")
        with open(cfg_path) as f:
            override = yaml.safe_load(f)
        for key, val in override.items():
            if key == "_base_":
                continue
            if isinstance(val, dict) and key in cfg:
                cfg[key].update(val)
            else:
                cfg[key] = val
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--config", default="small")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    np.random.seed(args.seed)

    from env.warehouse_env import WarehouseEnv
    from train import make_agent, run_episode

    env = WarehouseEnv(cfg["env"], seed=args.seed)
    agent = make_agent(args.agent, cfg, env)

    if args.checkpoint and os.path.exists(args.checkpoint):
        agent.load(args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")

    all_metrics = []
    for ep in range(args.n_eval_episodes):
        metrics = run_episode(env, agent, train=False)
        all_metrics.append(metrics)

    # Aggregate
    keys = all_metrics[0].keys()
    summary = {k: (np.mean([m[k] for m in all_metrics]), np.std([m[k] for m in all_metrics])) for k in keys}

    print(f"\n=== Eval: {args.agent} / {args.config} / seed={args.seed} ===")
    for k, (mean, std) in summary.items():
        print(f"  {k:30s}: {mean:8.3f} ± {std:.3f}")

    if not args.no_wandb:
        from utils.wandb_logger import WandbLogger
        logger = WandbLogger(
            agent_name=f"{args.agent}_eval", scenario=args.config, seed=args.seed
        )
        logger.log_episode({k: v[0] for k, v in summary.items()}, episode=0)
        logger.finish()


if __name__ == "__main__":
    main()
