"""
REINFORCE trainer — vanilla policy gradient with optional baseline.

Algorithm
---------
  For each step t in the episode:
    G_t = sum_{k=t}^{T} gamma^(k-t) * r_k          (discounted return)
    A_t = G_t - b_t                                  (advantage)
    loss = -mean( A_t * log_prob_t ) - beta * entropy_t

Baseline options
----------------
  "none"   : A_t = G_t  (high variance, simple)
  "mean"   : A_t = G_t - mean(G)  (reduce variance with no extra params)
  "value"  : A_t = G_t - V(s_t)  (requires ActorCriticPolicy)
             Also adds a value loss term: MSE(V(s_t), G_t)

Compatible policies: MLPPolicy, ActorCriticPolicy, GNNPolicy
"""

import numpy as np
import torch
import torch.optim as optim

from agents.trainers.base_trainer import BaseTrainer
from agents.policies.base_policy import BasePolicy


class REINFORCETrainer(BaseTrainer):
    """
    Parameters
    ----------
    lr            : learning rate
    gamma         : discount factor
    entropy_coef  : entropy bonus weight (encourages exploration)
    value_coef    : value loss weight (only used with "value" baseline)
    baseline      : "none" | "mean" | "value"
    """

    def __init__(
        self,
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        baseline: str = "mean",
    ):
        self.gamma        = gamma
        self.entropy_coef = entropy_coef
        self.value_coef   = value_coef
        self.baseline     = baseline
        self._lr          = lr
        self._optimizer   = None   # created lazily on first update

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_optimizer(self, policy: BasePolicy) -> optim.Optimizer:
        if self._optimizer is None:
            self._optimizer = optim.Adam(policy.parameters(), lr=self._lr)
        return self._optimizer

    @staticmethod
    def _compute_returns(rewards: list, gamma: float) -> torch.Tensor:
        """Compute discounted returns G_t for each step t."""
        T       = len(rewards)
        returns = torch.zeros(T)
        G       = 0.0
        for t in reversed(range(T)):
            G          = rewards[t] + gamma * G
            returns[t] = G
        return returns

    # ------------------------------------------------------------------
    # BaseTrainer interface
    # ------------------------------------------------------------------

    def update(self, batch: dict, policy: BasePolicy) -> dict:
        rewards     = batch["rewards"]
        raw_actions = batch["raw_actions"]   # list of (N*N,) arrays
        encoded_obs = batch["encoded_obs"]

        T       = len(rewards)
        returns = self._compute_returns(rewards, self.gamma)   # (T,)

        # Collate observations and actions into tensors
        obs_batch = policy.collate_obs(encoded_obs)
        act_batch = torch.FloatTensor(np.stack(raw_actions))   # (T, N*N)

        # Recompute log_probs and entropies with current policy parameters
        log_probs, entropies, values = policy.evaluate_actions(obs_batch, act_batch)

        # --- Baseline ---------------------------------------------------
        if self.baseline == "value" and values is not None:
            advantages = returns - values.detach()
            value_loss = torch.nn.functional.mse_loss(values, returns)
        elif self.baseline == "mean":
            advantages = returns - returns.mean()
            value_loss = torch.tensor(0.0)
        else:
            advantages = returns
            value_loss = torch.tensor(0.0)

        # Normalise advantages for stability
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- Policy gradient loss ---------------------------------------
        pg_loss      = -(log_probs * advantages).mean()
        entropy_loss = -entropies.mean()
        loss         = pg_loss + self.entropy_coef * entropy_loss + self.value_coef * value_loss

        opt = self._get_optimizer(policy)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        opt.step()

        return {
            "policy_loss": pg_loss.item(),
            "entropy":     -entropy_loss.item(),
            "value_loss":  value_loss.item() if isinstance(value_loss, torch.Tensor) else 0.0,
            "loss":        loss.item(),
        }

    def state_dict(self) -> dict:
        if self._optimizer is None:
            return {}
        return {"optimizer": self._optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        if self._optimizer is not None and "optimizer" in state:
            self._optimizer.load_state_dict(state["optimizer"])
