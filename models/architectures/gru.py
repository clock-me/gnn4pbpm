import typing as tp
import torch
from torch import dropout, nn
from models.time2vec import Time2Vec

class GRUEmbedder(nn.Module):
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
        # ^-- has shape[max_len, batch_size, hidden_size]
        output, _ = self.gru(event_activities_embeddings)  # shape[L, N, d * H]
        return output 


class GRUByTokenClassifier(nn.Module):
    def __init__(
        self,
        num_activities: int,
        hidden_size: int,
        num_gru_layers: int,
        num_classes: int,
        time2vec_k: tp.Optional[int] = None,
        dropout=0.1,
    ) -> None:
        super().__init__()
        self.gru_embedder = GRUEmbedder(
            num_activities,
            hidden_size,
            num_gru_layers,
            time2vec_k,
            dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(
        self,
        event_activities: torch.Tensor,
        event_times: tp.Optional[torch.Tensor] = None,
    ): 
        embeddings = self.gru_embedder(event_activities, event_times)
        logits = self.classifier(embeddings)
        return logits 

class GRUClassifier(nn.Module):
    def __init__(
        self,
        num_activities: int,
        hidden_size: int,
        num_gru_layers: int,
        num_classes: int,
        time2vec_k: tp.Optional[bool],
        dropout=0.1
    ) -> None:
        super().__init__()
        self.embedder = GRUEmbedder(
            num_activities,
            hidden_size,
            num_gru_layers,
            time2vec_k,
            dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(
        self,
        event_activities: torch.Tensor,
        event_times: tp.Optional[torch.Tensor] = None,
    ): 
        embeddings = self.embedder(event_activities, event_times)
        output = torch.mean(embeddings, dim=0)
        logits = self.classifier(output)
        return logits 

    
class GRURegressor(nn.Module):
    def __init__(
        self,
        num_activities: int,
        hidden_size: int,
        num_gru_layers: int,
        time2vec_k: int,
        dropout=0.1,
    ) -> None:
        super().__init__()
        self.embedder = GRUEmbedder(
            num_activities,
            hidden_size,
            num_gru_layers,
            time2vec_k,
            dropout=dropout
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        event_activities: torch.Tensor,
        event_times: torch.Tensor,
    ): 
        embeddings = self.embedder(event_activities, event_times)
        logits = self.regressor(embeddings)
        return logits.squeeze(-1) 
