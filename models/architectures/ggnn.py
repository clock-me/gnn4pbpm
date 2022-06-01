from audioop import add
from torch import nn 
import torch_geometric.nn as tg_nn
import torch_scatter
import torch
from torch.nn import functional as F

from models.architectures import edge_embedder

class GGNNEmbedder(nn.Module):
    def __init__(
        self,
        num_activities: int,
        num_timestamps: int,
        hidden_size: int,
        add_freq_features: bool,
        do_pooling: bool = True
    ) -> None:
        super().__init__()
        self.node_embedding = nn.Embedding(
            num_activities,
            embedding_dim=hidden_size,
        )
        self.ggnn = tg_nn.GatedGraphConv(
            out_channels=hidden_size,
            num_layers=num_timestamps,
        )
        self.use_edge_features = add_freq_features
        self.do_pooling = do_pooling
        if add_freq_features:
            self.edge_embedder = edge_embedder.EdgeEmbedder(
                add_time_features=False,
                add_freq_features=True,
                add_type_features=False,
            )


    def forward(self, graph):
        node_embeddings = self.node_embedding(graph.x.squeeze(-1))
        if self.use_edge_features:
            edge_embeddings = self.edge_embedder(graph.edge_attr)
            node_embeddings = self.ggnn(node_embeddings, graph.edge_index, edge_embeddings)
        else:
            node_embeddings = self.ggnn(node_embeddings, graph.edge_index)
        # ^-- shape[n_nodes, hidden_size]
        if self.do_pooling:
            node_embeddings = torch_scatter.scatter_mean(
                node_embeddings,
                graph.batch, 
                dim=0
            )  # shape [batch_size, hidden_size]
        return node_embeddings
    
class GGNNClassifier(nn.Module):
    def __init__(
        self,
        num_activities: int,
        num_timestamps: int,
        hidden_size: int,
        num_classes: int,
        add_freq_features: bool
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            GGNNEmbedder(
                num_activities,
                num_timestamps,
                hidden_size,
                add_freq_features=add_freq_features,
            ),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes), 
        )
    
    def forward(self, graph):
        return self.model(graph)

class GGNNByTokenClassifier(GGNNClassifier):
    def forward(self, sequential_graph, lens):
        logits = self.model(sequential_graph) # shape[sum(lens), num_activities]
        mx_len = torch.max(lens)
        splitted = torch.split(logits, tuple(lens.tolist()), dim=0)
        logits = torch.stack([
            F.pad(splitted[i], (0, 0, 0, mx_len - splitted[i].shape[0]))
            for i in range(len(splitted))
        ]).permute(1, 0, 2)
        return logits

class GGNNRegressor(nn.Module):
    def __init__(
        self,
        num_activities: int,
        num_timestamps: int,
        hidden_size: int,
        add_freq_features: bool,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            GGNNEmbedder(
                num_activities,
                num_timestamps,
                hidden_size,
                add_freq_features=add_freq_features,
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