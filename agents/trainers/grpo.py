"""
GRPO trainer — Group Relative Policy Optimization (DeepSeek-R1 style).

Algorithm (DeepSeek-R1, Shao et al. 2024)
------------------------------------------
  Instead of a learned value baseline, GRPO uses a GROUP of G rollouts
  collected from similar starting states to estimate advantages:

    For a group of G episodes with returns {G_1, ..., G_G}:
      A_i = (G_i - mean(G)) / (std(G) + eps)

  No critic / value network needed. Advantages are entirely relative
  to the other episodes in the group.

  For each episode in the group, standard clipped PPO objective is applied:
    L = -mean( min(r*A, clip(r, 1-eps, 1+eps)*A) ) - beta * entropy

Implementation note
-------------------
  The trainer buffers episodes until G are accumulated, then performs
  one gradient update over the full group. Returns {} on all intermediate
  calls (no update yet).

  The "same initial state" requirement from the paper is relaxed:
  we group consecutive G episodes. In practice this works well since
  the policy changes slowly between episodes.

Compatible policies: MLPPolicy, GNNPolicy (no value head needed)
NOT for ActorCriticPolicy (its value head is unused and wasteful here —
use PPO instead if you want a value baseline).
"""

import numpy as np
import torch
import torch.optim as optim

from agents.trainers.base_trainer import BaseTrainer
from agents.policies.base_policy import BasePolicy


class GRPOTrainer(BaseTrainer):
    """
    Parameters
    ----------
    lr           : learning rate
    gamma        : discount factor (for computing episode returns)
    clip_eps     : PPO clipping epsilon
    group_size   : G — number of episodes per group (typically 4–16)
    n_epochs     : gradient epochs per group update
    entropy_coef : entropy bonus coefficient
    eps          : small constant for advantage normalisation
    """

    def __init__(
        self,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        group_size: int = 8,
        n_epochs: int = 1,
        entropy_coef: float = 0.0,
        eps: float = 1e-8,
    ):
        self.gamma        = gamma
        self.clip_eps     = clip_eps
        self.group_size   = group_size
        self.n_epochs     = n_epochs
        self.entropy_coef = entropy_coef
        self.eps          = eps
        self._lr          = lr
        self._optimizer   = None

        # Episode buffer: accumulate until group_size episodes
        self._buffer: list[dict] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_optimizer(self, policy: BasePolicy) -> optim.Optimizer:
        if self._optimizer is None:
            self._optimizer = optim.Adam(policy.parameters(), lr=self._lr)
        return self._optimizer

    @staticmethod
    def _episode_return(rewards: list, gamma: float) -> float:
        """Discounted total return of an episode."""
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
        return G

    @staticmethod
    def _mc_returns(rewards: list, gamma: float) -> torch.Tensor:
        """Per-step discounted returns G_t."""
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
        # Accumulate episode into buffer
        self._buffer.append(batch)

        # Wait until the group is full
        if len(self._buffer) < self.group_size:
            return {}

        # --- Group is ready: compute group-relative advantages ----------
        group_returns = [
            self._episode_return(ep["rewards"], self.gamma)
            for ep in self._buffer
        ]
        group_mean = float(np.mean(group_returns))
        group_std  = float(np.std(group_returns)) + self.eps

        # Build a flat dataset across the whole group
        all_obs, all_acts, all_advs, all_old_lps = [], [], [], []

        for ep, ep_return in zip(self._buffer, group_returns):
            T       = len(ep["rewards"])
            ep_adv  = (ep_return - group_mean) / group_std   # scalar advantage for this episode

            # Per-step advantage: same scalar for every step (episode-level GRPO)
            # This is the simplest variant; step-level GRPO uses per-step returns.
            step_advs = torch.full((T,), ep_adv, dtype=torch.float32)

            all_obs.extend(ep["encoded_obs"])
            all_acts.append(torch.FloatTensor(np.stack(ep["raw_actions"])))
            all_advs.append(step_advs)
            all_old_lps.append(torch.FloatTensor(ep["log_probs"]))

        all_acts   = torch.cat(all_acts,    dim=0)   # (N_total, action_dim)
        all_advs   = torch.cat(all_advs,    dim=0)   # (N_total,)
        all_old_lps = torch.cat(all_old_lps, dim=0)  # (N_total,)

        opt = self._get_optimizer(policy)
        clip_losses, entropies = [], []

        for _ in range(self.n_epochs):
            obs_batch = policy.collate_obs(all_obs)
            log_probs, ent, _ = policy.evaluate_actions(obs_batch, all_acts)

            ratio    = (log_probs - all_old_lps).exp()
            surr1    = ratio * all_advs
            surr2    = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * all_advs
            clip_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -ent.mean()
            loss = clip_loss + self.entropy_coef * entropy_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            opt.step()

            clip_losses.append(clip_loss.item())
            entropies.append(-entropy_loss.item())

        # Clear buffer for next group
        self._buffer.clear()

        return {
            "policy_loss":    float(np.mean(clip_losses)),
            "entropy":        float(np.mean(entropies)),
            "group_mean_ret": group_mean,
            "group_std_ret":  group_std,
        }

    def state_dict(self) -> dict:
        state = {"buffer_len": len(self._buffer)}
        if self._optimizer is not None:
            state["optimizer"] = self._optimizer.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        if self._optimizer is not None and "optimizer" in state:
            self._optimizer.load_state_dict(state["optimizer"])
