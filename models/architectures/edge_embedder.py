from sympy import O
from torch import nn
from models import time2vec
import torch


class EdgeEmbedder(nn.Module):
    def __init__(
        self,
        add_time_features: bool = True,
        add_freq_features: bool = True,
        add_type_features: bool = True,
        time_features_k: int = 20,
        emb_size: int = 20,
    ) -> None:
        super().__init__()
        self.add_time_features = add_time_features
        self.add_freq_features = add_freq_features
        self.add_type_features = add_type_features
        assert add_time_features or add_freq_features or add_type_features
        self.edge_dim = 0
        if add_freq_features:
            self.edge_dim += 1
        if add_time_features:
            self.time2vec = time2vec.Time2Vec(
                time_features_k
            ) 
            self.edge_dim += time_features_k + 1
        if add_type_features:
            self.type_embeddings = nn.Embedding(
                2,
                emb_size,
            )
            self.edge_dim += emb_size
        

    def forward(self, edge_attrs):
        result = []
        if self.add_time_features:
            result.append(self.time2vec(edge_attrs[:, 2]))
        if self.add_freq_features:
            weights = torch.softmax(edge_attrs[:, 1], dim=0)
            weights = torch.masked_fill(weights, edge_attrs[:, 0] == 1, 1.0)
            result.append(weights.unsqueeze(-1))
        if self.add_type_features:
            result.append(self.type_embeddings(edge_attrs[:, 0].type(torch.int64))) 
        return torch.concat(result, dim=-1)
