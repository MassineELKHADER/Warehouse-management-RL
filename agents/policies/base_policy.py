"""
BasePolicy - abstract interface that every policy must implement.

Design goals
------------
1. Separation of obs encoding from network logic.
   `obs_extractor(state_dict) -> np.ndarray` converts the raw env state
   into the flat feature vector the network sees.
   Default: concat(inventory, demand) -> (2*N,).
   To add new features (e.g. "time", "forecast"), only change the extractor —
   zero changes to the policy network.

2. Consistent act / evaluate_actions contract so every trainer
   (REINFORCE, PPO, GRPO, SAC) works with every policy (MLP, AC, GNN).

3. Raw actions: policies return the *sample* from the distribution
   (BEFORE any projection). The env's _project_action() handles feasibility.
"""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch
import torch.nn as nn


def default_obs_extractor(state: dict) -> np.ndarray:
    """Default: flat concat of inventory and demand -> (2*N,)."""
    return np.concatenate([state["inventory"], state["demand"]])


def make_normalized_extractor(
    max_inventory: float,
    demand_scale: float,
) -> "Callable[[dict], np.ndarray]":
    """
    Return an obs_extractor that normalises inventory to [0, 1] and demand to
    [0, 1] (approximately) before feeding to the network.

    Parameters
    ----------
    max_inventory : env.max_inventory  (e.g. 10000)
    demand_scale  : upper bound for demand normalisation, e.g. 3 * demand_mean
    """
    def _extractor(state: dict) -> np.ndarray:
        inv = np.asarray(state["inventory"], dtype=np.float32) / max_inventory
        dem = np.asarray(state["demand"],    dtype=np.float32) / demand_scale
        return np.concatenate([inv, dem])
    return _extractor


class BasePolicy(ABC, nn.Module):
    """
    Abstract base for all policy networks.

    Subclasses must implement:
        act(state, deterministic) -> (flat_action, log_prob, entropy)
        evaluate_actions(obs_batch, actions_batch) -> (log_probs, entropies, values_or_None)

    Subclasses may override:
        encode_obs(state)  - default returns obs_extractor(state) as np.ndarray
        collate_obs(list)  - default stacks numpy arrays into a (B, obs_dim) tensor
    """

    def __init__(self, obs_extractor: Callable[[dict], np.ndarray] | None = None):
        nn.Module.__init__(self)
        self.obs_extractor: Callable[[dict], np.ndarray] = (
            obs_extractor or default_obs_extractor
        )

    # ------------------------------------------------------------------
    # Must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def act(
        self, state: dict, deterministic: bool = False
    ) -> tuple[np.ndarray, float, float]:
        """
        Sample an action for the given state.

        Returns
        -------
        flat_action : (N*N,) np.ndarray - raw distribution sample BEFORE projection
        log_prob    : float - log probability of the sampled action
        entropy     : float - policy entropy at this state
        """
        ...

    @abstractmethod
    def evaluate_actions(
        self,
        obs_batch,               # output of collate_obs()
        actions_batch: torch.Tensor,  # (B, N*N) raw flat actions
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Recompute (log_probs, entropies, values) for a stored batch.
        Called during the training update.

        Returns
        -------
        log_probs : (B,) tensor
        entropies : (B,) tensor
        values    : (B,) tensor  - only for ActorCriticPolicy; None otherwise
        """
        ...

    # ------------------------------------------------------------------
    # May override
    # ------------------------------------------------------------------

    def encode_obs(self, state: dict):
        """
        Encode a state dict for later batching.
        Default: apply obs_extractor -> np.ndarray.
        GNNPolicy overrides this to return the raw state dict (graph is built at batch time).
        """
        return self.obs_extractor(state)

    def collate_obs(self, obs_list: list) -> torch.Tensor:
        """
        Collate a list of encoded observations into a batch for evaluate_actions.
        Default: stack numpy arrays -> (B, obs_dim) float tensor.
        GNNPolicy overrides this to batch graphs via PyG.
        """
        return torch.FloatTensor(np.stack(obs_list))
