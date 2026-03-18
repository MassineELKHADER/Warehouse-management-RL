from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """
    Common interface all agents must implement.

    The shared train.py / evaluate.py only call act(), update(), save(), load().
    Internals (SB3, custom PyTorch, GNN, ...) are fully up to each subcomponent.
    """

    @abstractmethod
    def act(self, state: dict) -> np.ndarray:
        """
        Given the current state, return a transport matrix T of shape (N, N).
        T[i, j] = quantity to ship from warehouse i to warehouse j.
        Must satisfy: sum_j T[i,j] <= state['inventory'][i] for all i.
        """

    @abstractmethod
    def update(self, batch: dict) -> dict:
        """
        Perform one gradient / policy update given a batch of experience.

        batch keys (all numpy arrays):
            states      : list of state dicts (or stacked arrays)
            actions     : (B, N, N)
            rewards     : (B,)
            next_states : list of state dicts
            dones       : (B,) bool

        Returns a dict of scalar metrics, e.g. {"loss": 0.42, "entropy": 1.3}.
        Return {} if this agent has no learning step (e.g. baselines).
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model weights / state to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights / state from disk."""
