from torch import nn
import torch
from torch.nn import functional as F
import typing as tp


from models.architectures import gat
from models.architectures import gru
from models.time2vec import Time2Vec

class BaseEmbedder(nn.Module):
    def __init__(
        self,
        num_activities: int,
        hidden_size: int,
        num_gru_layers: int,
        time2vec_k: tp.Optional[int] = None,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.activity_embedding = nn.Embedding(
            num_embeddings=num_activities,
            embedding_dim=hidden_size,
        )
        adding = 0 if time2vec_k is None else time2vec_k + 1
        self.gru = nn.GRU(
            input_size=hidden_size + adding,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            dropout=dropout,
        ) 
        if time2vec_k is not None:
            self.time_embedder = Time2Vec(time2vec_k)

    def forward(
        self,
        event_activities: torch.Tensor,
        event_times: tp.Optional[torch.Tensor] = None,
    ): 
        """
        event_activities has shape[max_len, batch_size]
        event_times has shape[max_len, batch_size]
        """
        event_activities_embeddings = self.activity_embedding(event_activities)
        if event_times is not None:
            time_embeddings = self.time_embedder(event_times)
            event_activities_embeddings = torch.cat([time_embeddings, event_activities_embeddings], dim=-1)
        return event_activities_embeddings
        # ^-- has shape[max_len, batch_size, hidden_size]
        output, _ = self.gru(event_activities_embeddings)  # shape[L, N, d * H]
        return output 

class GATGRUEmbedder(nn.Module):
    def __init__(
        self,
        num_activities,
        hidden_size,
        num_gru_layers, 
        num_node_types,
        num_gat_heads,
        num_gat_layers,
    ) -> None:
        super().__init__()
        self.base_embedder = BaseEmbedder(
            num_activities=num_activities,
            hidden_size=hidden_size,
            num_gru_layers=num_gru_layers,
            time2vec_k=20,
        )
        self.gat_embedder = gat.GATEmbedder(
            num_activities=num_node_types,
            num_heads=num_gat_heads,
            num_layers=num_gat_layers,
            hidden_size=hidden_size,
            add_time_features=True,
            add_freq_features=True,
            add_type_features=True,
            do_pooling=True,
        )
        self.gru = nn.GRU(
            input_size=hidden_size + 21 + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            dropout=0.1,
        )  
    
    def forward(
        self,
        graph,
        activities_sequence,
        time_sequence,
        seq_graph_lengths,
    ):
        gat_embs = self.gat_embedder(graph)
        mx_len = torch.max(seq_graph_lengths)
        splitted = torch.split(gat_embs, tuple(seq_graph_lengths.tolist()), dim=0)
        gat_embs = torch.stack([
            F.pad(splitted[i], (0, 0, 0, mx_len - splitted[i].shape[0]))
            for i in range(len(splitted))
        ]).permute(1, 0, 2)
        base_embs = self.base_embedder(activities_sequence, time_sequence)
        embs_before_gru = torch.cat([gat_embs, base_embs], dim=-1)
        gru_embs, _ = self.gru(embs_before_gru)
        return gru_embs
    
class GATGRUByTokenClassifier(nn.Module):
    def __init__(
        self,
        num_activities,
        hidden_size,
        num_gru_layers, 
        num_node_types,
        num_gat_heads,
        num_gat_layers,
        num_classes,
    ) -> None:
        super().__init__()
        self.embedder = GATGRUEmbedder(
            num_activities,
            hidden_size,
            num_gru_layers,
            num_node_types,
            num_gat_heads,
            num_gat_layers,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(
        self,
        graph,
        seq_graph_lengths,
        activities_sequence,
        time_sequence
    ):
        embeddings = self.embedder(
            graph,
            activities_sequence,
            time_sequence,
            seq_graph_lengths,
        )
        logits = self.classifier(embeddings)
        return logits
