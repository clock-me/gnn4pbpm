
from torch_geometric.data import Data
import torch

def make_sequential_graph(num_edges, max_nodes=4):
    cur_edges = []
    x = torch.arange(max_nodes)
    result = []
    for i in range(num_edges):
        cur_edges.append([i, i + 1])
        edge_index = torch.tensor(cur_edges).T 
        data = Data(x=x, edge_index=edge_index)
        result.append(data)
    return result

def make_graph(num_edges, max_nodes=4):
    cur_edges = []
    x = torch.arange(max_nodes)
    for i in range(num_edges):
        cur_edges.append([i, i + 1])
    edge_index = torch.tensor(cur_edges).T 
    data = Data(x=x, edge_index=edge_index)
    return data
