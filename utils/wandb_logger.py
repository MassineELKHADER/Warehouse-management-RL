"""
Thin WandB wrapper that enforces standardised metric names across all agents.

Usage:
    logger = WandbLogger(agent_name="ppo", scenario="small", seed=42)
    logger.log_step({"episode_reward": -42.0, "total_transport_cost": 12.0, ...})
    logger.log_episode(metrics_dict, episode=5)
    logger.finish()
"""

import os
from dotenv import load_dotenv

load_dotenv()

REQUIRED_METRICS = {
    "episode_reward",
    "total_transport_cost",
    "demand_satisfaction",
    "inventory_std",
}


class WandbLogger:
    def __init__(self, agent_name: str, scenario: str, seed: int):
        import wandb

        self.run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "warehouse-rl"),
            entity=os.getenv("WANDB_ENTITY", None),
            name=f"{agent_name}_{scenario}_{seed}",
            group=agent_name,
            tags=[agent_name, scenario, f"seed_{seed}"],
            config={
                "agent": agent_name,
                "scenario": scenario,
                "seed": seed,
            },
        )

    def log_episode(self, metrics: dict, episode: int) -> None:
        missing = REQUIRED_METRICS - metrics.keys()
        if missing:
            raise ValueError(f"WandbLogger: missing required metrics: {missing}")
        self.run.log({"episode": episode, **metrics})

    def log(self, data: dict, step: int | None = None) -> None:
        """Generic log for training-step level metrics (loss, entropy, etc.)."""
        self.run.log(data, step=step)

    def finish(self) -> None:
        self.run.finish()
