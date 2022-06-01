from mimetypes import add_type
from torch import nn
from torch.nn import functional as F
import torch_geometric
from torch_geometric.nn.models import GAT
import torch_scatter
import torch
import typing as tp

from models.architectures import edge_embedder

class GATEmbedder(nn.Module):
    def __init__(
        self,
        num_activities: int,
        num_layers: int,
        num_heads: int,
        hidden_size: int,
        do_pooling: int,
        add_time_features: bool = False,
        add_freq_features: bool = True,
        add_type_features: bool = False,
    ) -> None:
        super().__init__()
        self.node_embedding = nn.Embedding(
            num_activities,
            embedding_dim=hidden_size,
        )
        self.use_edge_features = add_type_features or add_freq_features or add_time_features
        if add_time_features or add_freq_features or add_type_features:
            self.edge_embedder = edge_embedder.EdgeEmbedder(
                add_time_features,
                add_freq_features,
                add_type_features,
                emb_size=hidden_size,
            )
            edge_dim = self.edge_embedder.edge_dim
        else:
            edge_dim = None
        self.gat = GAT(
            in_channels=hidden_size,
            hidden_channels=hidden_size,
            num_layers=num_layers,
            heads=num_heads,
            edge_dim=edge_dim
        )
        self.do_pooling = do_pooling

    def forward(self, graph):
        node_embeddings = self.node_embedding(graph.x.squeeze(-1))
        if self.use_edge_features:
            edge_embeddings = self.edge_embedder(graph.edge_attr)
            node_embeddings = self.gat(node_embeddings, graph.edge_index, edge_embeddings)
        else:
            node_embeddings = self.gat(node_embeddings, graph.edge_index)
            
        if self.do_pooling:
            node_embeddings = torch_scatter.scatter_mean(
                node_embeddings,
                graph.batch, 
                dim=0
            )  # shape [batch_size, hidden_size] 
        return node_embeddings

class GATClassifier(nn.Module):
    def __init__(
        self,
        num_activities: int,
        num_layers: int,
        num_heads: int,
        hidden_size: int,
        num_classes: int,
        add_time_features: bool = False,
        add_freq_features: bool = True,
        add_type_features: bool = False,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            GATEmbedder(
                num_activities,
                num_layers,
                num_heads,
                hidden_size,
                do_pooling=True,
                add_time_features=add_time_features,
                add_freq_features=add_freq_features,
                add_type_features=add_type_features,
            ),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes), 
        )
    
    def forward(self, graph):
        return self.model(graph)

class GATByTokenClassifier(GATClassifier):
    def forward(self, sequential_graph, lens):
        logits = self.model(sequential_graph) # shape[sum(lens), num_activities]
        mx_len = torch.max(lens)
        splitted = torch.split(logits, tuple(lens.tolist()), dim=0)
        logits = torch.stack([
            F.pad(splitted[i], (0, 0, 0, mx_len - splitted[i].shape[0]))
            for i in range(len(splitted))
        ]).permute(1, 0, 2)
        return logits

class GATRegressor(nn.Module):
    def __init__(
        self,
        num_activities: int,
        num_layers: int,
        num_heads: int,
        hidden_size: int,
        add_time_features: bool = False,
        add_freq_features: bool = True,
        add_type_features: bool = False,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            GATEmbedder(
                num_activities,
                num_layers,
                num_heads,
                hidden_size,
                do_pooling=True,
                add_time_features=add_time_features,
                add_freq_features=add_freq_features,
                add_type_features=add_type_features,
            ),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1), 
        )
    
    def forward(self, sequential_graph):
        logits = []
        for graph in sequential_graph:
            cur_logits = self.model(graph).squeeze(1) # shape[batch_size]
            logits.append(cur_logits) 
        return torch.stack(logits)