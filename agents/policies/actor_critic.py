"""
ActorCriticPolicy - shared MLP backbone with a policy head and a value head.

Architecture
------------
  obs (2*N) -> shared_net -> mean_head (N*N)   ← actor
                          + log_std  (N*N)
                          -> value_head (1)    ← critic

Use this policy with trainers that need a value estimate:
  - REINFORCE  : value head used as variance-reducing baseline
  - PPO        : value head used for GAE advantage estimation

Not useful with GRPO (no critic needed) or SAC (has its own Q structure).
"""

from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from agents.policies.base_policy import BasePolicy


class ActorCriticPolicy(BasePolicy):
    """
    Shared-backbone actor-critic.

    Parameters
    ----------
    obs_dim       : input dimension
    action_dim    : N*N flat action dimension
    hidden        : hidden layer width
    obs_extractor : optional feature extractor callable
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

        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
        )
        # Actor head
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std   = nn.Parameter(torch.zeros(action_dim))
        # Critic head
        self.value_head = nn.Linear(hidden, 1)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.shared(obs)

    def _dist(self, features: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mean_head(features)
        std  = self.log_std.clamp(-4.0, 0.5).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    # ------------------------------------------------------------------
    # Extra: value estimate (used by REINFORCE baseline and PPO/GAE)
    # ------------------------------------------------------------------

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return scalar value estimate V(obs). Shape: (B,) or scalar."""
        return self.value_head(self._features(obs)).squeeze(-1)

    # ------------------------------------------------------------------
    # BasePolicy interface
    # ------------------------------------------------------------------

    def act(
        self, state: dict, deterministic: bool = False
    ) -> tuple[np.ndarray, float, float]:
        obs = torch.FloatTensor(self.obs_extractor(state)).unsqueeze(0)
        with torch.no_grad():
            feat     = self._features(obs)
            dist     = self._dist(feat)
            action   = dist.mean if deterministic else dist.rsample()
            log_prob = dist.log_prob(action).sum(-1).item()
            entropy  = dist.entropy().sum(-1).item()
        return action.squeeze(0).numpy(), log_prob, entropy

    def evaluate_actions(
        self,
        obs_batch: torch.Tensor,        # (B, obs_dim)
        actions_batch: torch.Tensor,    # (B, action_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat      = self._features(obs_batch)
        dist      = self._dist(feat)
        log_probs = dist.log_prob(actions_batch).sum(-1)   # (B,)
        entropies = dist.entropy().sum(-1)                  # (B,)
        values    = self.value_head(feat).squeeze(-1)       # (B,)
        return log_probs, entropies, values
