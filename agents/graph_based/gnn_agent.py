"""
P3 — GNN Agent skeleton.

Architecture:
  - Input graph: nodes = warehouses (features: inventory, demand)
                 edges = shipping routes (features: cost)
  - GNN backbone: GATConv or SAGEConv from PyTorch Geometric
  - Output: per-edge score → softmax per source node → transport matrix T

Training can be done with:
  - REINFORCE (simplest)
  - PPO on top of the GNN policy (recommended for stability)

TODO (P3):
  - Choose GNN layer type (GATConv recommended — attends to edge features)
  - Decide on number of layers and hidden dim
  - Implement update() with your chosen policy gradient method
  - Add a value head if using PPO
"""

import numpy as np
import torch
import torch.nn as nn
from agents.base_agent import BaseAgent
from agents.graph_based.graph_builder import build_graph, graph_to_action

try:
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False


class GNNPolicy(nn.Module):
    """
    2-layer GAT that produces a scalar score for each directed edge.

    Node features: (N, 2)  → [inventory, demand]
    Edge features: (E, 1)  → [cost]
    Output:        (E,)    → edge scores (before softmax per source node)
    """

    def __init__(self, node_feat_dim: int = 2, edge_feat_dim: int = 1, hidden: int = 64, heads: int = 4):
        super().__init__()
        if not _PYG_AVAILABLE:
            raise ImportError("torch_geometric is required for GNNPolicy. Install with: pip install torch-geometric")

        self.conv1 = GATConv(node_feat_dim, hidden, heads=heads, edge_dim=edge_feat_dim, concat=True)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1, edge_dim=edge_feat_dim, concat=False)
        # Edge scoring: concatenate source + dest node embeddings → scalar
        self.edge_score = nn.Sequential(
            nn.Linear(hidden * 2 + edge_feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        h = torch.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        h = self.conv2(h, edge_index, edge_attr=edge_attr)  # (N, hidden)

        # Gather source and dest embeddings for each edge
        src_h = h[edge_index[0]]   # (E, hidden)
        dst_h = h[edge_index[1]]   # (E, hidden)
        edge_input = torch.cat([src_h, dst_h, edge_attr], dim=-1)  # (E, 2*hidden + 1)
        scores = self.edge_score(edge_input).squeeze(-1)  # (E,)
        return scores


class GNNAgent(BaseAgent):
    """
    Graph Neural Network policy for warehouse redistribution.

    TODO (P3):
    - Implement update() with REINFORCE or PPO
    - Add entropy regularisation
    - Optionally add a GNN value head for variance reduction
    """

    def __init__(
        self,
        cost_matrix: np.ndarray,
        hidden: int = 64,
        heads: int = 4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.cost_matrix = cost_matrix
        self.n = cost_matrix.shape[0]
        self.gamma = gamma
        self.device = torch.device(device)

        self.policy = GNNPolicy(hidden=hidden, heads=heads).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def act(self, state: dict) -> np.ndarray:
        graph = build_graph(state, self.cost_matrix)
        x = graph["x"].to(self.device)
        edge_index = graph["edge_index"].to(self.device)
        edge_attr = graph["edge_attr"].to(self.device)

        with torch.no_grad():
            scores = self.policy(x, edge_index, edge_attr)  # (E,)

        T = graph_to_action(scores.cpu(), state["inventory"], self.n)
        return T

    def update(self, batch: dict) -> dict:
        # TODO: implement policy gradient update
        # batch should contain: graphs, actions, rewards, log_probs, ...
        raise NotImplementedError("GNN update not yet implemented — P3 TODO")

    def save(self, path: str) -> None:
        torch.save({"policy": self.policy.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
