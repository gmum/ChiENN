from collections import defaultdict

import torch
import torch_geometric
import torch_geometric.data
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, to_undirected

from chienn.data.edge_graph.get_circle_index import get_circle_index


def to_edge_graph(data: Data) -> Data:
    """
    Converts the graph to a graph of edges. Every directed edge (a, b) with index i becomes a node (denoted with node')
    with attribute of the form data.x[a] | data.edge_attr[i]. Then every node' (x, a) incoming to node a is connected
    with node' (a, b) with directed edge'. For compatibility with GINE, edge_attr' of edge' (a, b) -> (b, c) are set
    to data.edge_attr[j], where j is the index of edge (a, b).

    Args:
        data: torch geometric data with nodes attributes (x), edge attributes (edge_attr) and edge indices (edge_index)

    Returns:
        Graph of edges.
    """

    if not is_undirected(data.edge_index):
        edge_index, edge_attr = to_undirected(edge_index=data.edge_index, edge_attr=data.edge_attr)
    else:
        edge_index, edge_attr = data.edge_index, data.edge_attr

    new_nodes = []
    new_nodes_to_idx = {}
    for edge, edge_attr in zip(edge_index.T, edge_attr):
        a, b = edge
        a, b = a.item(), b.item()
        a2b = torch.cat([data.x[a], edge_attr, data.x[b]])  # x_{i, j} = x'_i | e'_{i, j} | x'_j.
        pos = torch.cat([data.pos[a], data.pos[b]])
        new_nodes_to_idx[(a, b)] = len(new_nodes)
        new_nodes.append(
            {'a': a, 'b': b, 'a_attr': data.x[a], 'node_attr': a2b, 'old_edge_attr': edge_attr, 'pos': pos})

    in_nodes = defaultdict(list)
    for i, node_dict in enumerate(new_nodes):
        a, b = node_dict['a'], node_dict['b']
        in_nodes[b].append({'node_idx': i, 'start_node_idx': a})

    new_edges = []
    for i, node_dict in enumerate(new_nodes):
        a, b = node_dict['a'], node_dict['b']
        ab_old_edge_attr = node_dict['old_edge_attr']
        a_attr = node_dict['a_attr']
        a_in_nodes_indices = [d['node_idx'] for d in in_nodes[a]]
        for in_node_c in a_in_nodes_indices:
            in_node = new_nodes[in_node_c]
            ca_old_edge_attr = in_node['old_edge_attr']
            # e_{(i, j), (j, k)} = e'_(i, j) | x'_j | e'_{k, j}:
            edge_attr = torch.cat([ca_old_edge_attr, a_attr, ab_old_edge_attr])
            new_edges.append({'edge': [in_node_c, i], 'edge_attr': edge_attr})

    parallel_node_index = []
    for node_dict in new_nodes:
        a, b = node_dict['a'], node_dict['b']
        parallel_idx = new_nodes_to_idx[(b, a)]
        parallel_node_index.append(parallel_idx)

    new_x = [d['node_attr'] for d in new_nodes]
    new_pos = [d['pos'] for d in new_nodes]
    new_edge_index = [d['edge'] for d in new_edges]
    new_edge_attr = [d['edge_attr'] for d in new_edges]
    new_x = torch.stack(new_x)
    new_pos = torch.stack(new_pos)
    new_edge_index = torch.tensor(new_edge_index).T
    new_edge_attr = torch.stack(new_edge_attr)
    parallel_node_index = torch.tensor(parallel_node_index)

    data = torch_geometric.data.Data(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr, pos=new_pos)
    data.parallel_node_index = parallel_node_index
    data.circle_index = get_circle_index(data, clockwise=False)
    return data
