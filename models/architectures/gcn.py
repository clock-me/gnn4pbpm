from mimetypes import add_type
from turtle import forward
from torch import nn
from torch_geometric.nn.models import GCN
import torch_scatter
import torch
from torch.nn import functional as F


from models.architectures import edge_embedder

class GCNEmbedder(nn.Module):
    def __init__(
        self,
        num_activities: int,
        num_layers: int,
        hidden_size: int,
        do_pooling: int,
        add_freq_features: bool=False,
    ) -> None:
        super().__init__()
        self.node_embedding = nn.Embedding(
            num_activities,
            embedding_dim=hidden_size,
        )
        self.use_edge_features = add_freq_features
        if add_freq_features:
            self.edge_embedding = edge_embedder.EdgeEmbedder(
                add_freq_features=True,
                add_time_features=False,
                add_type_features=False,
            )
        self.gcn = GCN(
            in_channels=hidden_size,
            hidden_channels=hidden_size,
            num_layers=num_layers,
            dropout=0.1
        )
        self.do_pooling = do_pooling

    def forward(self, graph):
        node_embeddings = self.node_embedding(graph.x.squeeze(-1))
        if self.use_edge_features:
            edge_weights = self.edge_embedding(graph.edge_attr)
            edge_weights = edge_weights.squeeze(-1) 
            node_embeddings = self.gcn(node_embeddings, graph.edge_index, edge_weights)
        else:
            node_embeddings = self.gcn(node_embeddings, graph.edge_index)
        if self.do_pooling:
            node_embeddings = torch_scatter.scatter_mean(
                node_embeddings,
                graph.batch, 
                dim=0
            )  # shape [batch_size, hidden_size] 
        return node_embeddings

class GCNClassifier(nn.Module):
    def __init__(
        self,
        num_activities: int,
        num_layers: int,
        hidden_size: int,
        num_classes: int,
        add_freq_features: bool,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            GCNEmbedder(
                num_activities,
                num_layers,
                hidden_size,
                do_pooling=True,
                add_freq_features=add_freq_features,
            ),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes), 
        )
    
    def forward(self, graph):
        return self.model(graph)

class GCNByTokenClassifier(GCNClassifier):
    def forward(self, sequential_graph, lens):
        logits = self.model(sequential_graph) # shape[sum(lens), num_activities]
        mx_len = torch.max(lens)
        splitted = torch.split(logits, tuple(lens.tolist()), dim=0)
        logits = torch.stack([
            F.pad(splitted[i], (0, 0, 0, mx_len - splitted[i].shape[0]))
            for i in range(len(splitted))
        ]).permute(1, 0, 2)
        return logits

class GCNRegressor(nn.Module):
    def __init__(
        self,
        num_activities: int,
        num_layers: int,
        hidden_size: int,
        add_freq_features: bool,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            GCNEmbedder(
                num_activities,
                num_layers,
                hidden_size,
                do_pooling=True,
                add_freq_features=add_freq_features
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