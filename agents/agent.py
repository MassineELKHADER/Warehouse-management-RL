"""
Agent - combines any BasePolicy with any BaseTrainer into a BaseAgent.

This is the glue layer between the modular policy/trainer system and the
existing train.py / evaluate.py entry points, which only call:
    agent.act(state)   -> (N, N) transport matrix
    agent.update(batch) -> metrics dict
    agent.save(path)
    agent.load(path)

How trajectory data flows
--------------------------
train.py calls act() once per step and collects a batch dict with keys:
    states, actions, rewards, next_states, dones

act() additionally stores internally:
    raw_actions   - the raw distribution sample BEFORE projection
                    (needed for log_prob recomputation in PPO/REINFORCE/GRPO)
    log_probs     - log prob at sample time (used as old_log_probs in PPO/GRPO)
    entropies     - entropy at each step
    encoded_obs   - policy.encode_obs(state) for batching at update time

update() merges the external batch with the internally stored data and
passes the augmented batch to trainer.update(batch, policy).

Usage
-----
    from agents.policies import MLPPolicy, ActorCriticPolicy, GNNPolicy
    from agents.trainers import PPOTrainer, GRPOTrainer, REINFORCETrainer, SACTrainer

    policy  = ActorCriticPolicy(obs_dim=2*N, action_dim=N*N)
    trainer = PPOTrainer(lr=3e-4, gamma=0.99)
    agent   = Agent(policy, trainer, n_warehouses=N)

    # Swap in any combination:
    agent = Agent(GNNPolicy(cost_matrix), REINFORCETrainer(baseline="mean"), N)
    agent = Agent(MLPPolicy(obs_dim, N*N), GRPOTrainer(group_size=8), N)
"""

import numpy as np
import torch

from agents.base_agent import BaseAgent
from agents.policies.base_policy import BasePolicy
from agents.trainers.base_trainer import BaseTrainer


class Agent(BaseAgent):
    """
    Policy × Trainer wrapper implementing the BaseAgent interface.

    Parameters
    ----------
    policy       : any BasePolicy subclass
    trainer      : any BaseTrainer subclass
    n_warehouses : N - needed to reshape flat action -> (N, N) matrix
    """

    def __init__(self, policy: BasePolicy, trainer: BaseTrainer, n_warehouses: int):
        self._policy    = policy
        self._trainer   = trainer
        self._n         = n_warehouses
        self._trajectory: list[dict] = []   # cleared after each update()

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, state: dict, deterministic: bool = False) -> np.ndarray:
        """
        Sample action from policy and store trajectory data for update().

        Returns the (N, N) transport matrix. The env's _project_action()
        will clip/scale it to satisfy all constraints.
        """
        flat_action, log_prob, entropy = self._policy.act(state, deterministic)

        # Store trajectory data needed by the trainer
        self._trajectory.append({
            "raw_action":  flat_action,                    # (N*N,) pre-projection
            "log_prob":    log_prob,
            "entropy":     entropy,
            "encoded_obs": self._policy.encode_obs(state), # for batching
        })

        # Pass raw action to env; _project_action() clips negatives to 0
        # and enforces all other constraints (diagonal=0, row sums ≤ inventory).
        T = flat_action.reshape(self._n, self._n)
        return T

    def update(self, batch: dict) -> dict:
        """
        Augment the external batch with trajectory data, call trainer.update(),
        then clear the trajectory buffer.
        """
        if not self._trajectory:
            return {}

        aug_batch = dict(batch)
        aug_batch["raw_actions"]  = [t["raw_action"]  for t in self._trajectory]
        aug_batch["log_probs"]    = [t["log_prob"]     for t in self._trajectory]
        aug_batch["entropies"]    = [t["entropy"]      for t in self._trajectory]
        aug_batch["encoded_obs"]  = [t["encoded_obs"]  for t in self._trajectory]

        metrics = self._trainer.update(aug_batch, self._policy)
        self._trajectory.clear()
        return metrics

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy":  self._policy.state_dict(),
                "trainer": self._trainer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self._policy.load_state_dict(ckpt["policy"])
        if "trainer" in ckpt:
            self._trainer.load_state_dict(ckpt["trainer"])

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def policy(self) -> BasePolicy:
        return self._policy

    @property
    def trainer(self) -> BaseTrainer:
        return self._trainer
