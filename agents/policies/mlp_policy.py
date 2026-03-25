"""
MLPPolicy - Gaussian policy over the flattened (N*N) transport matrix.

Architecture
------------
  obs (2*N) -> Linear -> Tanh -> Linear -> Tanh -> mean_head (N*N)
                                              + log_std  (N*N, learnable param)

The policy outputs a factored Gaussian N(mean, diag(std²)) over the flat
action space. Actions are sampled and passed raw to the env; _project_action()
enforces feasibility (non-negative, no self-shipment, inventory limits).

Compatible trainers: REINFORCE, PPO, GRPO, SAC.
"""

from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from agents.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy):
    """
    Two-hidden-layer MLP Gaussian policy.

    Parameters
    ----------
    obs_dim       : input dimension (default 2*N for inventory + demand)
    action_dim    : output dimension (N*N - flat transport matrix)
    hidden        : hidden layer width
    obs_extractor : optional callable to change what features are fed in
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: int = 128,
        obs_extractor: Callable | None = None,
    ):
        super().__init__(obs_extractor)
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        # Global learnable log std (one per action dimension)
        self.log_std   = nn.Parameter(torch.zeros(action_dim))

    # ------------------------------------------------------------------
    # Internal: distribution
    # ------------------------------------------------------------------

    def _dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        h    = self.net(obs)
        mean = self.mean_head(h)
        std  = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    # ------------------------------------------------------------------
    # BasePolicy interface
    # ------------------------------------------------------------------

    def act(
        self, state: dict, deterministic: bool = False
    ) -> tuple[np.ndarray, float, float]:
        obs = torch.FloatTensor(self.obs_extractor(state)).unsqueeze(0)
        with torch.no_grad():
            dist = self._dist(obs)
            action = dist.mean if deterministic else dist.rsample()
            log_prob = dist.log_prob(action).sum(-1).item()
            entropy  = dist.entropy().sum(-1).item()
        return action.squeeze(0).numpy(), log_prob, entropy

    def evaluate_actions(
        self,
        obs_batch: torch.Tensor,        # (B, obs_dim)
        actions_batch: torch.Tensor,    # (B, action_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        dist      = self._dist(obs_batch)
        log_probs = dist.log_prob(actions_batch).sum(-1)   # (B,)
        entropies = dist.entropy().sum(-1)                  # (B,)
        return log_probs, entropies, None
