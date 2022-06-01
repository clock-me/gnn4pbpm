
import pytorch_lightning as pl
from torch.nn import functional as F
from constants import *

from models import base_tasks
from models.architectures import transformer



class TransformerForNAP(base_tasks.BaseNAP):
    def __init__(
        self,
        num_activities,
        hidden_size,
        num_heads,
        num_layers,
        dropout,
        learning_rate,
        max_timestamps=256,
        time2vec_k=None,
    ):
        super().__init__(learning_rate)
        self.use_time2vec = time2vec_k is not None
        self.model = transformer.TransformerByTokenClassifier(
            num_activities,
            hidden_size,
            max_timestamps,
            num_heads,
            num_layers,
            dropout,
            num_activities,
            time2vec_k=time2vec_k
        )
    def forward(self, batch):
        if not self.use_time2vec:
            return self.model(
                batch[ACTIVITIES_SEQUENCE][:-1],
                batch[PADDING_MASK][:-1],
            )
        else:
            return self.model(
                batch[ACTIVITIES_SEQUENCE][:-1],
                batch[PADDING_MASK][:-1],
                batch[TIME_SEQUENCE][:-1],
            )


class TransformerForBOP(base_tasks.BaseBOP):
    def __init__(
        self,
        num_activities,
        hidden_size,
        num_heads,
        num_layers,
        dropout,
        learning_rate,
        max_timestamps=256,
        time2vec_k=None,
    ):
        super().__init__(learning_rate)
        self.use_time2vec = time2vec_k is not None
        self.model = transformer.TransformerClassifier(
            num_activities=num_activities,
            hidden_size=hidden_size,
            max_timestamps=max_timestamps,
            nhead=num_heads,
            nlayers=num_layers,
            dropout=dropout,
            num_classes=2,
            time2vec_k=time2vec_k,
        )

    def forward(self, batch): 
        if not self.use_time2vec:
            return self.model(batch[ACTIVITIES_SEQUENCE], batch[PADDING_MASK])
        else:
            return self.model(batch[ACTIVITIES_SEQUENCE], batch[PADDING_MASK], batch[TIME_SEQUENCE])


class TransformerForNTP(pl.LightningModule):
    def __init__(
        self,
        num_activities,
        hidden_size,
        num_heads,
        num_layers,
        dropout,
        learning_rate,
        max_timestamps=256,
        time2vec_k=None,
    ):
        super().__init__(learning_rate)
        self.use_time2vec = time2vec_k is not None
        self.model = transformer.TransformerRegressor(
            num_activities=num_activities,
            hidden_size=hidden_size,
            nhead=num_heads,
            nlayers=num_layers,
            dropout=dropout, 
            max_timestamps=max_timestamps,
            time2vec_k=time2vec_k,
        )
    
    def forward(self, batch): 
        if not self.use_time2vec:
            return self.model(
                batch[ACTIVITIES_SEQUENCE][:-1],
                batch[PADDING_MASK][:-1],
            )
        else:
            return self.model(
                batch[ACTIVITIES_SEQUENCE][:-1],
                batch[PADDING_MASK][:-1],
                batch[TIME_SEQUENCE][:-1],
            )

    