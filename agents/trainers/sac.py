"""
SAC trainer — Soft Actor-Critic (off-policy, maximum-entropy RL).

Algorithm (Haarnoja et al. 2018)
---------------------------------
  Maintains:
    - Replay buffer of (obs, flat_action, reward, next_obs, done) tuples
    - Twin Q-networks Q1, Q2  (to reduce overestimation)
    - Target Q-networks Q1_target, Q2_target  (soft-updated)
    - Entropy temperature alpha (fixed or auto-tuned)

  Per update step:
    Critic: minimise  E[ (Q(s,a) - y)² ]
      where y = r + gamma*(1-d)*( min(Q1',Q2')(s',a') - alpha*log_pi(a'|s') )
    Actor:  minimise  E[ alpha*log_pi(a|s) - min(Q1,Q2)(s,a) ]
    Alpha:  minimise  E[ -alpha*(log_pi + target_entropy) ]  (if auto_alpha)

  update() adds one episode to the replay buffer and then performs
  `updates_per_episode` gradient steps.

Compatible policies: MLPPolicy only.
  - GNN+SAC is out of scope (Q over graphs is complex).
  - ActorCriticPolicy is not used (SAC has its own Q structure).
"""

from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.trainers.base_trainer import BaseTrainer
from agents.policies.base_policy import BasePolicy


# ---------------------------------------------------------------------------
# Twin Q-networks (internal to SAC, not a policy)
# ---------------------------------------------------------------------------

class _TwinQ(nn.Module):
    """Twin Q-networks that take (obs, flat_action) → scalar Q value."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        def _make_q():
            return nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        self.q1 = _make_q()
        self.q2 = _make_q()

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(obs, action)
        return torch.min(q1, q2)


# ---------------------------------------------------------------------------
# SAC Trainer
# ---------------------------------------------------------------------------

class SACTrainer(BaseTrainer):
    """
    Parameters
    ----------
    obs_dim            : flat observation dimension (2*N)
    action_dim         : flat action dimension (N*N)
    lr                 : learning rate (shared for actor, critic, alpha)
    gamma              : discount factor
    tau                : soft target update coefficient
    alpha              : initial entropy temperature (ignored if auto_alpha=True)
    auto_alpha         : learn entropy temperature automatically
    buffer_size        : maximum replay buffer capacity
    batch_size         : number of transitions per gradient step
    updates_per_episode: gradient steps to perform after each episode
    hidden             : Q-network hidden layer width
    warmup_steps       : minimum buffer size before starting updates
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_alpha: bool = True,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        updates_per_episode: int = 50,
        hidden: int = 256,
        warmup_steps: int = 1000,
    ):
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size
        self.updates_per_episode = updates_per_episode
        self.warmup_steps = warmup_steps

        # Replay buffer
        self._buffer: deque = deque(maxlen=buffer_size)
        self._total_steps = 0

        # Twin Q-networks
        self.critic        = _TwinQ(obs_dim, action_dim, hidden)
        self.critic_target = _TwinQ(obs_dim, action_dim, hidden)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        # Actor optimizer (created lazily to accept any BasePolicy)
        self._actor_opt = None
        self._lr        = lr

        # Entropy temperature
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -float(action_dim)   # heuristic: -dim(A)
            self.log_alpha      = torch.tensor(np.log(alpha), requires_grad=True, dtype=torch.float32)
            self.alpha_opt      = optim.Adam([self.log_alpha], lr=lr)
            self.alpha          = self.log_alpha.exp().item()
        else:
            self.alpha     = alpha
            self.log_alpha = None
            self.alpha_opt = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_actor_opt(self, policy: BasePolicy) -> optim.Optimizer:
        if self._actor_opt is None:
            self._actor_opt = optim.Adam(policy.parameters(), lr=self._lr)
        return self._actor_opt

    def _soft_update(self) -> None:
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

    def _add_to_buffer(self, batch: dict) -> None:
        """Add all transitions from one episode to the replay buffer."""
        T = len(batch["rewards"])
        for t in range(T):
            obs      = batch["encoded_obs"][t]                 # np.ndarray (obs_dim,)
            act      = np.abs(batch["raw_actions"][t])         # (N*N,) — abs as executed
            reward   = batch["rewards"][t]
            next_obs = (
                batch["encoded_obs"][t + 1]
                if t + 1 < T
                else batch["encoded_obs"][t]   # terminal: use same obs
            )
            done = batch["dones"][t]
            self._buffer.append((obs, act, reward, next_obs, done))
        self._total_steps += T

    def _sample_batch(self) -> tuple:
        transitions = random.sample(self._buffer, self.batch_size)
        obs, acts, rews, next_obs, dones = zip(*transitions)
        return (
            torch.FloatTensor(np.stack(obs)),
            torch.FloatTensor(np.stack(acts)),
            torch.FloatTensor(rews),
            torch.FloatTensor(np.stack(next_obs)),
            torch.FloatTensor(dones),
        )

    # ------------------------------------------------------------------
    # BaseTrainer interface
    # ------------------------------------------------------------------

    def update(self, batch: dict, policy: BasePolicy) -> dict:
        self._add_to_buffer(batch)

        if self._total_steps < self.warmup_steps:
            return {}

        actor_losses, critic_losses, alpha_losses, entropies = [], [], [], []
        actor_opt = self._get_actor_opt(policy)

        for _ in range(self.updates_per_episode):
            obs, acts, rews, next_obs, dones = self._sample_batch()

            # ---- Critic update -----------------------------------------
            with torch.no_grad():
                # Sample next action from current policy
                next_act_flat, next_lp, _ = zip(*[
                    policy.act({"inventory": next_obs[i, :self.obs_dim // 2].numpy(),
                                "demand":    next_obs[i, self.obs_dim // 2:].numpy()})
                    for i in range(self.batch_size)
                ])
                next_acts = torch.FloatTensor(np.stack([np.abs(a) for a in next_act_flat]))
                next_lps  = torch.FloatTensor(next_lp)

                q_next = self.critic_target.q_min(next_obs, next_acts)
                y      = rews + self.gamma * (1 - dones) * (q_next - self.alpha * next_lps)

            q1, q2       = self.critic(obs, acts)
            critic_loss  = nn.functional.mse_loss(q1, y) + nn.functional.mse_loss(q2, y)
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            # ---- Actor update ------------------------------------------
            # Re-sample actions differentiably
            obs_batch_t   = obs
            act_t_flat, lp_t, ent_t = zip(*[
                policy.act({"inventory": obs[i, :self.obs_dim // 2].numpy(),
                            "demand":    obs[i, self.obs_dim // 2:].numpy()})
                for i in range(self.batch_size)
            ])
            acts_t  = torch.FloatTensor(np.stack([np.abs(a) for a in act_t_flat]))
            lps_t   = torch.FloatTensor(lp_t)
            ents_t  = torch.FloatTensor(ent_t)

            q_pi        = self.critic.q_min(obs_batch_t, acts_t)
            actor_loss  = (self.alpha * lps_t - q_pi).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            actor_opt.step()

            # ---- Alpha update ------------------------------------------
            alpha_loss = torch.tensor(0.0)
            if self.auto_alpha:
                alpha_loss = -(self.log_alpha * (lps_t + self.target_entropy).detach()).mean()
                self.alpha_opt.zero_grad()
                alpha_loss.backward()
                self.alpha_opt.step()
                self.alpha = self.log_alpha.exp().item()

            self._soft_update()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            alpha_losses.append(alpha_loss.item())
            entropies.append(float(np.mean(ent_t)))

        return {
            "actor_loss":  float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "alpha_loss":  float(np.mean(alpha_losses)),
            "alpha":       self.alpha,
            "entropy":     float(np.mean(entropies)),
        }

    def state_dict(self) -> dict:
        state = {
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic_opt":    self.critic_opt.state_dict(),
            "alpha":         self.alpha,
            "total_steps":   self._total_steps,
        }
        if self._actor_opt is not None:
            state["actor_opt"] = self._actor_opt.state_dict()
        if self.auto_alpha:
            state["log_alpha"] = self.log_alpha.data
            state["alpha_opt"] = self.alpha_opt.state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        self.alpha       = state.get("alpha", self.alpha)
        self._total_steps = state.get("total_steps", 0)
        if self._actor_opt is not None and "actor_opt" in state:
            self._actor_opt.load_state_dict(state["actor_opt"])
        if self.auto_alpha and "log_alpha" in state:
            self.log_alpha.data.copy_(state["log_alpha"])
            self.alpha_opt.load_state_dict(state["alpha_opt"])
