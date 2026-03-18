"""
Shared training entry point.

Usage:
    python train.py --agent random  --config small  --seed 42
    python train.py --agent greedy  --config small  --seed 42
    python train.py --agent ppo     --config medium --seed 123
    python train.py --agent sac     --config medium --seed 123
    python train.py --agent gnn     --config large  --seed 7
"""

import argparse
import os
import yaml
import numpy as np
from dotenv import load_dotenv

load_dotenv()


def load_config(config_name: str) -> dict:
    base_path = os.path.join("configs", "default.yaml")
    cfg_path = os.path.join("configs", f"{config_name}.yaml")

    with open(base_path) as f:
        cfg = yaml.safe_load(f)

    if config_name != "default":
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


def make_agent(agent_name: str, cfg: dict, env):
    obs_dim = env.obs_dim
    n = env.n

    if agent_name == "random":
        from agents.baselines.random_agent import RandomAgent
        return RandomAgent()

    elif agent_name == "greedy":
        from agents.baselines.greedy_agent import GreedyAgent
        return GreedyAgent(cost_matrix=env.cost_matrix)

    elif agent_name == "ppo":
        from agents.model_free.ppo_agent import PPOAgent
        return PPOAgent(obs_dim=obs_dim, n_warehouses=n, gamma=cfg["env"]["gamma"])

    elif agent_name == "sac":
        from agents.model_free.sac_agent import SACAgent
        return SACAgent(obs_dim=obs_dim, n_warehouses=n, gamma=cfg["env"]["gamma"])

    elif agent_name == "gnn":
        from agents.graph_based.gnn_agent import GNNAgent
        return GNNAgent(cost_matrix=env.cost_matrix, gamma=cfg["env"]["gamma"])

    else:
        raise ValueError(f"Unknown agent: {agent_name}")


def run_episode(env, agent, train: bool = True) -> dict:
    from utils.metrics import episode_summary

    state = env.reset()
    done = False
    rewards, costs, satisfactions, inventories = [], [], [], []
    batch = {"states": [], "actions": [], "rewards": [], "next_states": [], "dones": []}

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        rewards.append(reward)
        costs.append(info["transport_cost"])
        satisfactions.append(info["demand_satisfaction"])
        inventories.append(info["inventory"])

        if train:
            batch["states"].append(state)
            batch["actions"].append(action)
            batch["rewards"].append(reward)
            batch["next_states"].append(next_state)
            batch["dones"].append(done)

        state = next_state

    update_metrics = {}
    if train:
        try:
            update_metrics = agent.update(batch)
        except NotImplementedError:
            pass  # baselines have no update

    metrics = episode_summary(rewards, costs, satisfactions, inventories)
    metrics.update(update_metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=["random", "greedy", "ppo", "sac", "gnn"])
    parser.add_argument("--config", default="small", choices=["default", "small", "medium", "large"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    np.random.seed(args.seed)

    from env.warehouse_env import WarehouseEnv
    env = WarehouseEnv(cfg["env"], seed=args.seed)
    agent = make_agent(args.agent, cfg, env)

    logger = None
    if not args.no_wandb:
        from utils.wandb_logger import WandbLogger
        logger = WandbLogger(agent_name=args.agent, scenario=args.config, seed=args.seed)

    n_episodes = cfg["training"]["n_episodes"]
    eval_every = cfg["training"]["eval_every"]

    print(f"Training {args.agent} on {args.config} scenario (seed={args.seed})")
    for ep in range(1, n_episodes + 1):
        metrics = run_episode(env, agent, train=True)

        if ep % eval_every == 0:
            print(f"  ep {ep:4d} | reward={metrics['episode_reward']:8.2f} | "
                  f"cost={metrics['total_transport_cost']:7.2f} | "
                  f"sat={metrics['demand_satisfaction']:.3f}")
            if logger:
                logger.log_episode(metrics, episode=ep)

    # Save checkpoint
    os.makedirs("outputs", exist_ok=True)
    ckpt_path = f"outputs/{args.agent}_{args.config}_{args.seed}.pt"
    try:
        agent.save(ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")
    except Exception:
        pass

    if logger:
        logger.finish()


if __name__ == "__main__":
    main()
