"""
P2 — SAC Agent skeleton.

SAC (Soft Actor-Critic) is well-suited for continuous action spaces.
You can implement from scratch or wrap SB3:

  from stable_baselines3 import SAC as SB3_SAC

The skeleton below is a from-scratch starting point with TODOs.
"""

import numpy as np
import torch
import torch.nn as nn
from agents.base_agent import BaseAgent
from agents.model_free.utils import project_action, flat_to_matrix

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def forward(self, x):
        h = self.net(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)


class Critic(nn.Module):
    """Twin Q-networks for SAC."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        def make_q():
            return nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        self.q1 = make_q()
        self.q2 = make_q()

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


class SACAgent(BaseAgent):
    """
    SAC with squashed Gaussian policy.

    TODO (P2):
    - Implement replay buffer
    - Implement update() with actor + twin-critic + temperature losses
    - Add target network with soft updates
    - Tune alpha (entropy temperature) — optionally make it learnable
    """

    def __init__(
        self,
        obs_dim: int,
        n_warehouses: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        device: str = "cpu",
    ):
        self.n = n_warehouses
        self.action_dim = n_warehouses ** 2
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = torch.device(device)

        self.actor = Actor(obs_dim, self.action_dim).to(self.device)
        self.critic = Critic(obs_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(obs_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def act(self, state: dict) -> np.ndarray:
        obs = np.concatenate([state["inventory"], state["demand"]])
        x = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor.sample(x)
        flat = action.squeeze(0).cpu().numpy()
        T = flat_to_matrix(np.abs(flat), self.n)
        T = project_action(T, state["inventory"])
        return T

    def update(self, batch: dict) -> dict:
        # TODO: implement SAC update
        # Expected batch keys: obs, actions, rewards, next_obs, dones
        raise NotImplementedError("SAC update not yet implemented — P2 TODO")

    def save(self, path: str) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
