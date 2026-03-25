"""
PPO trainer — Proximal Policy Optimization with clipped surrogate objective.

Algorithm
---------
  1. Collect one episode (done by Agent)
  2. Compute GAE advantages using value estimates (requires ActorCriticPolicy
     or falls back to Monte-Carlo returns for value-free policies)
  3. For n_epochs: randomly sample mini-batches and compute:
       L_clip  = -mean( min(r*A, clip(r, 1-eps, 1+eps)*A) )
       L_value = coef_v * MSE(V(s), V_target)
       L_ent   = -coef_e * mean(entropy)
       loss    = L_clip + L_value + L_ent
  4. Gradient step with global norm clipping

Compatible policies: MLPPolicy, ActorCriticPolicy, GNNPolicy
  - With ActorCriticPolicy: full GAE with learned value baseline
  - With MLPPolicy / GNNPolicy: Monte-Carlo returns as advantage (no value head)
"""

import numpy as np
import torch
import torch.optim as optim

from agents.trainers.base_trainer import BaseTrainer
from agents.policies.base_policy import BasePolicy


class PPOTrainer(BaseTrainer):
    """
    Parameters
    ----------
    lr          : learning rate
    gamma       : discount factor
    gae_lambda  : GAE smoothing parameter (1.0 = MC returns, 0.0 = TD)
    clip_eps    : PPO clipping epsilon
    n_epochs    : number of gradient epochs per episode
    batch_size  : mini-batch size (None = full batch)
    value_coef  : value loss coefficient
    entropy_coef: entropy bonus coefficient
    """

    def __init__(
        self,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        n_epochs: int = 10,
        batch_size: int | None = 64,
        value_coef: float = 0.5,
        entropy_coef: float = 0.001,
    ):
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.clip_eps     = clip_eps
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.value_coef   = value_coef
        self.entropy_coef = entropy_coef
        self._lr          = lr
        self._optimizer   = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_optimizer(self, policy: BasePolicy) -> optim.Optimizer:
        if self._optimizer is None:
            self._optimizer = optim.Adam(policy.parameters(), lr=self._lr)
        return self._optimizer

    def _compute_gae(
        self,
        rewards: list,
        values: torch.Tensor | None,   # (T,) value estimates, or None
        dones: list,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (advantages, value_targets).
        Falls back to Monte-Carlo if no value estimates are available.
        """
        T = len(rewards)
        if values is None:
            # Monte-Carlo returns — no value head
            returns    = torch.zeros(T)
            G          = 0.0
            for t in reversed(range(T)):
                G          = rewards[t] + self.gamma * G * (1.0 - float(dones[t]))
                returns[t] = G
            advantages = returns.clone()
        else:
            # GAE with learned value baseline
            vals_np = values.detach().numpy()
            adv     = np.zeros(T, dtype=np.float32)
            gae     = 0.0
            for t in reversed(range(T)):
                next_val = 0.0 if dones[t] else (vals_np[t + 1] if t + 1 < T else 0.0)
                delta    = rewards[t] + self.gamma * next_val - vals_np[t]
                gae      = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * gae
                adv[t]   = gae
            advantages = torch.FloatTensor(adv)
            returns    = advantages + values.detach()
        return advantages, returns

    # ------------------------------------------------------------------
    # BaseTrainer interface
    # ------------------------------------------------------------------

    def update(self, batch: dict, policy: BasePolicy) -> dict:
        rewards     = batch["rewards"]
        dones       = batch["dones"]
        raw_actions = batch["raw_actions"]
        old_lps     = torch.FloatTensor(batch["log_probs"])   # (T,) collected at rollout time
        encoded_obs = batch["encoded_obs"]

        T           = len(rewards)
        obs_batch   = policy.collate_obs(encoded_obs)
        act_batch   = torch.FloatTensor(np.stack(raw_actions))  # (T, N*N)

        # --- Get bootstrap value estimates (for GAE) --------------------
        with torch.no_grad():
            _, _, values_init = policy.evaluate_actions(obs_batch, act_batch)

        advantages, returns = self._compute_gae(rewards, values_init, dones)

        # Normalise advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- PPO epochs -------------------------------------------------
        clip_losses, value_losses, entropies = [], [], []
        opt = self._get_optimizer(policy)

        for _ in range(self.n_epochs):
            # Mini-batch sampling
            indices = torch.randperm(T)
            bs      = self.batch_size or T
            for start in range(0, T, bs):
                idx = indices[start: start + bs]

                mb_obs  = (
                    [obs_batch[i] for i in idx.tolist()]
                    if isinstance(obs_batch, list)
                    else obs_batch[idx]
                )
                mb_act  = act_batch[idx]
                mb_adv  = advantages[idx]
                mb_ret  = returns[idx]
                mb_olp  = old_lps[idx]

                log_probs, ent, vals = policy.evaluate_actions(mb_obs, mb_act)

                # Clipped surrogate
                ratio    = (log_probs - mb_olp).exp()
                surr1    = ratio * mb_adv
                surr2    = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                clip_loss = -torch.min(surr1, surr2).mean()

                # Value loss (only if policy has value head)
                if vals is not None:
                    value_loss = torch.nn.functional.mse_loss(vals, mb_ret)
                else:
                    value_loss = torch.tensor(0.0)

                entropy_loss = -ent.mean()
                loss = clip_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                opt.step()

                clip_losses.append(clip_loss.item())
                value_losses.append(value_loss.item() if isinstance(value_loss, torch.Tensor) else 0.0)
                entropies.append(-entropy_loss.item())

        return {
            "policy_loss": float(np.mean(clip_losses)),
            "value_loss":  float(np.mean(value_losses)),
            "entropy":     float(np.mean(entropies)),
        }

    def state_dict(self) -> dict:
        if self._optimizer is None:
            return {}
        return {"optimizer": self._optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        if self._optimizer is not None and "optimizer" in state:
            self._optimizer.load_state_dict(state["optimizer"])
