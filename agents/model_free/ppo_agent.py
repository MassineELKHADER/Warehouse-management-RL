"""
P2 — PPO Agent skeleton.

Implement your PPO here. You have two paths:
  (A) Wrap Stable-Baselines3's PPO inside BaseAgent (quick start)
  (B) Write a full custom PyTorch PPO (full control)

Both are valid — just implement act() / update() / save() / load().

The template below shows approach (B) as a starting point with TODOs.
"""

import numpy as np
import torch
import torch.nn as nn
from agents.base_agent import BaseAgent
from agents.model_free.utils import project_action, flat_to_matrix


class PolicyNetwork(nn.Module):
    """
    Simple MLP that maps flattened state (2*N) → flat action (N*N).
    Replace or extend as needed (e.g. larger hidden layers, different activation).
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor):
        features = self.net(x)
        mean = self.mean_head(features)
        std = self.log_std.exp().expand_as(mean)
        return mean, std


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PPOAgent(BaseAgent):
    """
    PPO with Gaussian policy over flattened transport matrix.

    TODO (P2):
    - Fill in the update() method with PPO loss (clipped surrogate + value + entropy)
    - Add GAE computation
    - Add a replay buffer / rollout buffer
    - Tune hyperparameters (lr, clip_eps, n_epochs, batch_size, ...)
    """

    def __init__(
        self,
        obs_dim: int,
        n_warehouses: int,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        n_epochs: int = 10,
        device: str = "cpu",
    ):
        self.n = n_warehouses
        self.action_dim = n_warehouses ** 2
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.device = torch.device(device)

        self.policy = PolicyNetwork(obs_dim, self.action_dim).to(self.device)
        self.value_fn = ValueNetwork(obs_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_fn.parameters()), lr=lr
        )

    def act(self, state: dict) -> np.ndarray:
        obs = np.concatenate([state["inventory"], state["demand"]])
        x = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, std = self.policy(x)
            dist = torch.distributions.Normal(mean, std)
            flat_action = dist.sample().squeeze(0).cpu().numpy()
        T = flat_to_matrix(np.abs(flat_action), self.n)
        T = project_action(T, state["inventory"])
        return T

    def update(self, batch: dict) -> dict:
        # TODO: implement PPO update
        # Expected batch keys: states, actions, rewards, next_states, dones, log_probs, values
        raise NotImplementedError("PPO update not yet implemented — P2 TODO")

    def save(self, path: str) -> None:
        torch.save(
            {"policy": self.policy.state_dict(), "value": self.value_fn.state_dict()},
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.value_fn.load_state_dict(ckpt["value"])
