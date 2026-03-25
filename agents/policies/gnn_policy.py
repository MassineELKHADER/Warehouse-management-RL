"""
GNNPolicy - Graph Attention Network policy for warehouse redistribution.

Architecture
------------
  Graph (nodes=warehouses, edges=routes)
    node features : [inventory_i, demand_i]       shape (N, 2)
    edge features : [cost_ij]                     shape (N*(N-1), 1)
  -> GATConv x 2
  -> edge_score_head: concat(src_emb, dst_emb, edge_feat) -> mean_ij
  + global log_std (learnable, shared across all edges)

The policy outputs a factored Gaussian over the flat (N*N) action space,
where the mean for each edge i->j comes from the GNN and the diagonal is
forced to zero. This gives the same distribution interface as MLPPolicy,
so all trainers (REINFORCE, PPO, GRPO) work unchanged.

SAC is not supported (Q over graph observations is out of scope).

obs_extractor is not used by GNNPolicy - state is converted directly to a
graph via build_graph(). encode_obs / collate_obs are overridden accordingly.
"""

from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from agents.policies.base_policy import BasePolicy
from agents.graph_based.graph_builder import build_graph

try:
    from torch_geometric.nn import GATConv
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False


class GNNPolicy(BasePolicy):
    """
    GAT-based Gaussian policy.

    Parameters
    ----------
    cost_matrix   : (N, N) numpy array - fixed shipping costs (stored for graph building)
    hidden        : hidden embedding dimension
    heads         : number of attention heads in GATConv
    obs_extractor : ignored (GNN uses graph representation, not flat obs)
    """

    def __init__(
        self,
        cost_matrix: np.ndarray,
        hidden: int = 64,
        heads: int = 4,
        obs_extractor: Callable | None = None,   # kept for API parity, not used
    ):
        if not _PYG_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for GNNPolicy. "
                "Install with: pip install torch-geometric"
            )
        super().__init__(obs_extractor)

        self.cost_matrix = cost_matrix
        self.n           = cost_matrix.shape[0]
        self.action_dim  = self.n * self.n

        node_feat_dim = 2    # [inventory, demand]
        edge_feat_dim = 1    # [cost]

        # Two-layer GAT
        self.conv1 = GATConv(
            node_feat_dim, hidden, heads=heads,
            edge_dim=edge_feat_dim, concat=True
        )
        self.conv2 = GATConv(
            hidden * heads, hidden, heads=1,
            edge_dim=edge_feat_dim, concat=False
        )

        # Per-edge mean: concat(src_emb, dst_emb, edge_feat) -> scalar
        self.edge_score = nn.Sequential(
            nn.Linear(hidden * 2 + edge_feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # Global learnable log std (shared across all edges, like MLPPolicy)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _forward_graph(self, graph: dict) -> torch.Tensor:
        """
        Run the GNN on a single graph dict and return mean scores (N*N,).
        Diagonal entries (self-edges) are set to 0.
        """
        x          = graph["x"]           # (N, 2)
        edge_index = graph["edge_index"]  # (2, E)  E = N*(N-1)
        edge_attr  = graph["edge_attr"]   # (E, 1)

        h = torch.relu(self.conv1(x, edge_index, edge_attr=edge_attr))  # (N, hidden*heads)
        h = self.conv2(h, edge_index, edge_attr=edge_attr)               # (N, hidden)

        src_h      = h[edge_index[0]]                               # (E, hidden)
        dst_h      = h[edge_index[1]]                               # (E, hidden)
        edge_input = torch.cat([src_h, dst_h, edge_attr], dim=-1)  # (E, 2*hidden+1)
        scores     = self.edge_score(edge_input).squeeze(-1)        # (E,)

        # Scatter edge scores back into (N*N,) with zeros on diagonal
        mean_flat = torch.zeros(self.n * self.n)
        e = 0
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    mean_flat[i * self.n + j] = scores[e]
                    e += 1
        return mean_flat  # (N*N,)

    def _dist_from_graph(self, graph: dict) -> torch.distributions.Normal:
        mean = self._forward_graph(graph)
        std  = self.log_std.exp()
        return torch.distributions.Normal(mean, std)

    # ------------------------------------------------------------------
    # Override obs encoding (GNN uses graph, not flat vector)
    # ------------------------------------------------------------------

    def encode_obs(self, state: dict) -> dict:
        """Return the raw state dict - graph is built at batch time."""
        return state

    def collate_obs(self, obs_list: list) -> list:
        """
        Build graphs from stored state dicts and return as a list.
        Each graph is a dict with keys: x, edge_index, edge_attr.
        """
        return [build_graph(s, self.cost_matrix) for s in obs_list]

    # ------------------------------------------------------------------
    # BasePolicy interface
    # ------------------------------------------------------------------

    def act(
        self, state: dict, deterministic: bool = False
    ) -> tuple[np.ndarray, float, float]:
        graph = build_graph(state, self.cost_matrix)
        with torch.no_grad():
            dist     = self._dist_from_graph(graph)
            action   = dist.mean if deterministic else dist.rsample()
            log_prob = dist.log_prob(action).sum().item()
            entropy  = dist.entropy().sum().item()
        return action.numpy(), log_prob, entropy

    def evaluate_actions(
        self,
        obs_batch: list,                # list of graph dicts (from collate_obs)
        actions_batch: torch.Tensor,    # (B, N*N)
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        log_probs_list = []
        entropies_list = []
        for graph, action in zip(obs_batch, actions_batch):
            dist      = self._dist_from_graph(graph)
            log_probs_list.append(dist.log_prob(action).sum())
            entropies_list.append(dist.entropy().sum())
        log_probs = torch.stack(log_probs_list)  # (B,)
        entropies = torch.stack(entropies_list)  # (B,)
        return log_probs, entropies, None
