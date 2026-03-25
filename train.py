"""
Training entry point.

New-style (modular):
    python train.py --policy mlp       --trainer ppo       --config small  --seed 42
    python train.py --policy mlp       --trainer grpo      --config medium --seed 42
    python train.py --policy mlp       --trainer sac       --config small  --seed 42
    python train.py --policy ac        --trainer ppo       --config medium --seed 123
    python train.py --policy ac        --trainer reinforce --config small  --seed 7
    python train.py --policy gnn       --trainer ppo       --config large  --seed 42
    python train.py --policy gnn       --trainer reinforce --config small  --seed 42
    python train.py --policy gnn       --trainer grpo      --config medium --seed 7

Legacy shortcuts (--agent maps to a fixed policy+trainer pair):
    python train.py --agent random   --config small  --seed 42
    python train.py --agent greedy   --config small  --seed 42
    python train.py --agent ppo      --config medium --seed 42   → mlp + ppo
    python train.py --agent sac      --config medium --seed 42   → mlp + sac
    python train.py --agent gnn      --config large  --seed 42   → gnn + ppo
"""

import argparse
import os

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

_LEGACY_MAP = {
    "random": ("random",  None),
    "greedy": ("greedy",  None),
    "ppo":    ("mlp",     "ppo"),
    "sac":    ("mlp",     "sac"),
    "gnn":    ("gnn",     "ppo"),
}


def make_agent(policy_name: str, trainer_name: str | None, cfg: dict, env):
    """
    Instantiate an agent from a policy name + trainer name.

    policy_name  : "random" | "greedy" | "mlp" | "ac" | "gnn"
    trainer_name : "reinforce" | "ppo" | "grpo" | "sac" | None (baselines)
    """
    N       = env.n
    obs_dim = env.obs_dim        # 2*N
    act_dim = N * N
    gamma   = cfg["env"].get("gamma", 0.99)

    # --- Baselines (no trainer) -----------------------------------------
    if policy_name == "random":
        from agents.baselines.random_agent import RandomAgent
        return RandomAgent(), "random", "none"

    if policy_name == "greedy":
        from agents.baselines.greedy_agent import GreedyAgent
        return GreedyAgent(cost_matrix=env.cost_matrix), "greedy", "none"

    # --- Modular agents -------------------------------------------------
    from agents.agent import Agent
    from agents.policies import MLPPolicy, ActorCriticPolicy, GNNPolicy

    if policy_name == "mlp":
        policy = MLPPolicy(obs_dim=obs_dim, action_dim=act_dim)
    elif policy_name == "ac":
        policy = ActorCriticPolicy(obs_dim=obs_dim, action_dim=act_dim)
    elif policy_name == "gnn":
        policy = GNNPolicy(cost_matrix=env.cost_matrix)
    else:
        raise ValueError(f"Unknown policy: {policy_name!r}")

    if trainer_name == "reinforce":
        from agents.trainers import REINFORCETrainer
        baseline = "value" if policy_name == "ac" else "mean"
        trainer  = REINFORCETrainer(lr=3e-4, gamma=gamma, baseline=baseline)

    elif trainer_name == "ppo":
        from agents.trainers import PPOTrainer
        trainer = PPOTrainer(lr=3e-4, gamma=gamma)

    elif trainer_name == "grpo":
        from agents.trainers import GRPOTrainer
        trainer = GRPOTrainer(lr=3e-4, gamma=gamma, group_size=8)

    elif trainer_name == "sac":
        from agents.trainers import SACTrainer
        trainer = SACTrainer(obs_dim=obs_dim, action_dim=act_dim, gamma=gamma)

    else:
        raise ValueError(f"Unknown trainer: {trainer_name!r}")

    return Agent(policy, trainer, n_warehouses=N), policy_name, trainer_name


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, agent, train: bool = True) -> dict:
    from utils.metrics import episode_summary

    state = env.reset()
    done  = False
    rewards, costs, satisfactions, inventories, replenishments = [], [], [], [], []
    batch = {"states": [], "actions": [], "rewards": [], "next_states": [], "dones": []}

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        rewards.append(reward)
        costs.append(info["transport_cost"])
        satisfactions.append(info["demand_satisfaction"])
        inventories.append(info["inventory"])
        replenishments.append(info.get("replenishment_cost", 0.0))

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
            pass

    metrics = episode_summary(rewards, costs, satisfactions, inventories)
    metrics["replenishment_cost"] = float(np.sum(replenishments))
    metrics.update(update_metrics)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    # New-style args
    parser.add_argument("--policy",  default=None,
                        choices=["mlp", "ac", "gnn"])
    parser.add_argument("--trainer", default=None,
                        choices=["reinforce", "ppo", "grpo", "sac"])

    # Legacy shortcut
    parser.add_argument("--agent",   default=None,
                        choices=list(_LEGACY_MAP.keys()))

    parser.add_argument("--config",  default="small",
                        choices=["default", "small", "medium", "large"])
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    # Resolve policy / trainer names
    if args.agent:
        policy_name, trainer_name = _LEGACY_MAP[args.agent]
    elif args.policy:
        policy_name  = args.policy
        trainer_name = args.trainer
    else:
        parser.error("Provide either --agent or --policy + --trainer")

    cfg = load_config(args.config)
    np.random.seed(args.seed)

    from env.warehouse_env import WarehouseEnv
    env = WarehouseEnv(cfg, seed=args.seed)   # full cfg (not cfg["env"])

    agent, p_name, t_name = make_agent(policy_name, trainer_name, cfg, env)

    run_name = (
        f"{p_name}_{t_name}_{args.config}_s{args.seed}"
        if t_name != "none"
        else f"{p_name}_{args.config}_s{args.seed}"
    )

    logger = None
    if not args.no_wandb:
        from utils.wandb_logger import WandbLogger
        logger = WandbLogger(
            policy_name=p_name,
            trainer_name=t_name,
            scenario=args.config,
            seed=args.seed,
            cfg=cfg,
        )

    n_episodes = cfg["training"]["n_episodes"]
    eval_every  = cfg["training"]["eval_every"]

    print(f"Training [{run_name}] for {n_episodes} episodes")

    for ep in range(1, n_episodes + 1):
        metrics = run_episode(env, agent, train=True)

        if ep % eval_every == 0:
            print(
                f"  ep {ep:4d} | reward={metrics['episode_reward']:8.2f} | "
                f"cost={metrics['total_transport_cost']:7.2f} | "
                f"sat={metrics['demand_satisfaction']:.3f} | "
                f"replen={metrics['replenishment_cost']:.2f}"
            )
            if logger:
                logger.log_episode(metrics, episode=ep)

    # Save checkpoint
    os.makedirs("outputs", exist_ok=True)
    ckpt_path = f"outputs/{run_name}.pt"
    try:
        agent.save(ckpt_path)
        print(f"Saved: {ckpt_path}")
    except Exception:
        pass

    if logger:
        logger.finish()


if __name__ == "__main__":
    main()
