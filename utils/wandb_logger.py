"""
WandbLogger — standardised metric logging for all policy x trainer combinations.

Run naming convention:
    {policy}_{trainer}_{scenario}_s{seed}   e.g. mlp_ppo_small_s42
    {policy}_{scenario}_s{seed}             e.g. random_small_s42 (baselines)

Hyperparameters are logged as WandB config at init so every run is fully
reproducible and filterable from the WandB dashboard.
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
    def __init__(
        self,
        policy_name: str,
        trainer_name: str,
        scenario: str,
        seed: int,
        cfg: dict | None = None,
    ):
        import wandb

        # Build a human-readable run name
        if trainer_name and trainer_name != "none":
            run_name = f"{policy_name}_{trainer_name}_{scenario}_s{seed}"
            group    = f"{policy_name}_{trainer_name}"
        else:
            run_name = f"{policy_name}_{scenario}_s{seed}"
            group    = policy_name

        # Flatten hyperparams from cfg for easy comparison on WandB
        wandb_cfg = {
            "policy":        policy_name,
            "trainer":       trainer_name,
            "scenario":      scenario,
            "seed":          seed,
        }
        if cfg:
            env_cfg = cfg.get("env", {})
            wandb_cfg.update({
                "n_warehouses":    env_cfg.get("n_warehouses"),
                "lambda_penalty":  env_cfg.get("lambda_penalty"),
                "episode_length":  env_cfg.get("episode_length"),
                "demand_model":    env_cfg.get("demand_model"),
                "cost_matrix":     cfg.get("cost_matrix"),
                "ext_supplier":    env_cfg.get("external_supplier", {}).get("enabled", False),
                "ext_cost":        env_cfg.get("external_supplier", {}).get("cost_per_unit"),
                "n_episodes":      cfg.get("training", {}).get("n_episodes"),
            })

        self.run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "warehouse-rl"),
            entity=os.getenv("WANDB_ENTITY", None),
            name=run_name,
            group=group,
            tags=[policy_name, trainer_name, scenario, f"seed_{seed}"],
            config=wandb_cfg,
        )

    def log_episode(self, metrics: dict, episode: int) -> None:
        missing = REQUIRED_METRICS - metrics.keys()
        if missing:
            raise ValueError(f"WandbLogger: missing required metrics: {missing}")
        # Prefix all metrics with train/
        prefixed = {f"train/{k}": v for k, v in metrics.items()}
        prefixed["episode"] = episode
        self.run.log(prefixed)

    def log_eval(self, metrics: dict, episode: int) -> None:
        """Log evaluation metrics (prefixed with eval/)."""
        prefixed = {f"eval/{k}": v for k, v in metrics.items()}
        prefixed["episode"] = episode
        self.run.log(prefixed)

    def log(self, data: dict, step: int | None = None) -> None:
        """Generic log for arbitrary metrics (loss, alpha, etc.)."""
        self.run.log(data, step=step)

    def finish(self) -> None:
        self.run.finish()
