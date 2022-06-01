from torch import nn
import typing as tp
from models.time2vec import Time2Vec
import torch


class TransformerEmbedder(nn.Module):
    def __init__(
        self,
        num_activities: int,
        hidden_size: int,
        max_timestamps: int,
        nhead: int,
        nlayers: int,
        dropout: int,
        time2vec_k: tp.Optional[int] = None,
    ) -> None:
        super().__init__()
        adding = 0 if time2vec_k is None else time2vec_k + 1
        self.token_encoder = nn.Embedding(num_activities, hidden_size - adding)
        self.positional_encoder = nn.Embedding(max_timestamps, hidden_size)
        if time2vec_k is not None:
            self.time2vec = Time2Vec(time2vec_k)
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=2 * hidden_size,
            nhead=nhead,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, nlayers)

    def forward(
        self,
        activity_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        time_sequence: tp.Optional[torch.Tensor],
        masked: bool = True
    ):
        tokens_encoded = self.token_encoder(activity_ids)

        positions = torch.arange(len(activity_ids))\
            .repeat(activity_ids.shape[1], 1).T\
            .to(self.token_encoder.weight.device) 
        positions_encoded = self.positional_encoder(positions)

        embs = torch.cat([tokens_encoded, positions_encoded], dim=-1)

        if time_sequence is not None:
            time_encoded = self.time2vec(time_sequence) 
            embs = torch.cat([time_encoded, embs], dim=-1)
        
        if masked:
            attention_mask = (
                nn.Transformer.generate_square_subsequent_mask(len(activity_ids))
            ).to(self.positional_encoder.weight.device)
        else:
            attention_mask=None
        
        embs = self.transformer_encoder(
            embs,
            mask=attention_mask,
            src_key_padding_mask=torch.logical_not(padding_mask).T
        )
        return embs
    


class TransformerByTokenClassifier(nn.Module):
    def __init__(
        self,
        num_activities: int,
        hidden_size: int,
        max_timestamps: int,
        nhead: int,
        nlayers: int,
        dropout: int,
        num_classes: int,
        time2vec_k: tp.Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embedder = TransformerEmbedder(
            num_activities,
            hidden_size,
            max_timestamps,
            nhead,
            nlayers,
            dropout,
            time2vec_k
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Linear(2 * hidden_size, num_classes),
        )
    
    def forward(
        self,
        activity_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        time_sequence: tp.Optional[torch.Tensor] = None,
    ):
        embeddings = self.embedder(
            activity_ids,
            padding_mask,
            time_sequence,
            True,
        )
        logits = self.classifier(embeddings)
        return logits



class TransformerRegressor(nn.Module):
    def __init__(
        self,
        num_activities: int,
        hidden_size: int,
        max_timestamps: int,
        nhead: int,
        nlayers: int,
        dropout: int,
        time2vec_k: tp.Optional[int] = None
    ) -> None:
        super().__init__()
        self.embedder = TransformerEmbedder(
            num_activities,
            hidden_size,
            max_timestamps,
            nhead,
            nlayers,
            dropout,
            time2vec_k
        )
        self.regressor = nn.Sequential(
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Linear(2 * hidden_size, 1),
        )

    def forward(
        self,
        activity_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        time_sequence: tp.Optional[torch.Tensor] = None,
    ):
        embeddings = self.embedder(
            activity_ids,
            padding_mask,
            time_sequence,
            masked=True,
        )
        predictions = self.regressor(embeddings)
        return predictions

    
class TransformerClassifier(nn.Module):
    def __init__(
        self,
        num_activities: int,
        hidden_size: int,
        max_timestamps: int,
        nhead: int,
        nlayers: int,
        dropout: int,
        num_classes: int,
        time2vec_k: tp.Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embedder = TransformerEmbedder(
            num_activities,
            hidden_size,
            max_timestamps,
            nhead,
            nlayers,
            dropout,
            time2vec_k
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Linear(2 * hidden_size, num_classes),
        )

    def forward(
        self,
        activity_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        time_sequence: tp.Optional[torch.Tensor] = None,
    ):
        embeddings = self.embedder(
            activity_ids,
            padding_mask,
            time_sequence,
            masked=False,
        )
        # masked mean pooling
        embeddings = (embeddings * padding_mask[:, :, None]).sum(dim=0)\
             / padding_mask.sum(dim=0)[:, None]

        logits = self.classifier(embeddings)
        return logits