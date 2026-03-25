"""
BaseTrainer — abstract interface that every training algorithm must implement.

A trainer is stateful (holds optimizer, replay buffer, hyperparams) but is
completely decoupled from the policy network structure. It receives a rollout
batch + the current policy and returns a metrics dict.

Expected batch keys (populated by Agent.update):
    states       : list of raw state dicts
    actions      : list of (N,N) numpy arrays (projected, as executed)
    rewards      : list of floats
    next_states  : list of raw state dicts
    dones        : list of bools
    raw_actions  : list of (N*N,) numpy arrays (pre-projection distribution samples)
    log_probs    : list of floats (log prob at sample time)
    entropies    : list of floats
    encoded_obs  : list of encoded observations (from policy.encode_obs)
"""

from abc import ABC, abstractmethod
from agents.policies.base_policy import BasePolicy


class BaseTrainer(ABC):
    """
    Abstract training algorithm.

    Subclasses must implement update().
    Subclasses may implement state_dict() / load_state_dict() for checkpointing.
    """

    @abstractmethod
    def update(self, batch: dict, policy: BasePolicy) -> dict:
        """
        Perform one training update.

        Parameters
        ----------
        batch  : trajectory dict populated by Agent (see module docstring)
        policy : the policy network to update in-place

        Returns
        -------
        metrics : dict of scalar training metrics (loss, entropy, etc.)
                  Return {} if no update was performed (e.g. buffer not full).
        """
        ...

    def state_dict(self) -> dict:
        """Return trainer state for checkpointing (optimizer state, buffers, etc.)."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Restore trainer state from a checkpoint."""
        pass
