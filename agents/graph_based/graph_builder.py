"""
P3 — Graph Builder.

Converts a WarehouseEnv state dict + cost matrix into a PyTorch Geometric Data object.

Graph structure:
  Nodes: one per warehouse
    node_features: [inventory_i, demand_i]  shape (N, 2)

  Edges: fully connected (all i→j pairs, i≠j)
    edge_features: [cost_ij]                shape (N*(N-1), 1)
    edge_index: (2, N*(N-1))                COO format
"""

import numpy as np
import torch


def build_graph(state: dict, cost_matrix: np.ndarray):
    """
    Returns a dict with keys: x, edge_index, edge_attr
    Compatible with torch_geometric.data.Data(**graph).

    Args:
        state       : {"inventory": (N,), "demand": (N,)}
        cost_matrix : (N, N) numpy array
    """
    inventory = state["inventory"]   # (N,)
    demand = state["demand"]         # (N,)
    n = len(inventory)

    # Node features
    x = torch.tensor(
        np.stack([inventory, demand], axis=1), dtype=torch.float32
    )  # (N, 2)

    # Edge index and features (all i→j, i≠j)
    src, dst = [], []
    edge_feats = []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
                edge_feats.append([cost_matrix[i, j]])

    edge_index = torch.tensor([src, dst], dtype=torch.long)   # (2, E)
    edge_attr = torch.tensor(edge_feats, dtype=torch.float32)  # (E, 1)

    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}


def graph_to_action(edge_scores: torch.Tensor, inventory: np.ndarray, n: int) -> np.ndarray:
    """
    Convert per-edge scores output by GNN → transport matrix T (N, N).

    edge_scores : (E,) raw scores, one per directed edge i→j (i≠j)
    inventory   : (N,) current inventory (used to scale shipments)

    Strategy: softmax over outgoing edges of each node i,
    then multiply by inventory[i].
    """
    T = np.zeros((n, n), dtype=np.float32)
    edge_idx = 0
    for i in range(n):
        # Outgoing edges from i
        out_scores = []
        out_j = []
        for j in range(n):
            if i != j:
                out_scores.append(edge_scores[edge_idx].item())
                out_j.append(j)
                edge_idx += 1
        scores_t = torch.softmax(torch.tensor(out_scores), dim=0).numpy()
        for k, j in enumerate(out_j):
            T[i, j] = scores_t[k] * inventory[i]
    return T
