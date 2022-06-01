from torch import nn

import torchmetrics

import pytorch_lightning as pl
import torch
from constants import *

from models.architectures.gru import *
from models import base_tasks

class GRUForNAP(base_tasks.BaseNAP):
    def __init__(
        self,
        num_activities,
        hidden_size,
        num_layers,
        learning_rate,
        dropout=0.1,
        time2vec_k=None,
    ):
        super().__init__(learning_rate)
        self.use_time2vec = time2vec_k is not None
        self.model = GRUByTokenClassifier(
            num_activities=num_activities,
            num_classes=num_activities,
            hidden_size=hidden_size,
            num_gru_layers=num_layers,
            time2vec_k=time2vec_k,
            dropout=dropout,
        )
    
    def forward(self, batch):
        if not self.use_time2vec:
            logits = self.model(batch[ACTIVITIES_SEQUENCE][:-1])
        else:
            logits = self.model(batch[ACTIVITIES_SEQUENCE][:-1], batch[TIME_SEQUENCE][:-1])
        return logits

class GRUForNTP(base_tasks.BaseNTP):
    def __init__(
        self,
        num_activities,
        hidden_size,
        num_layers,
        learning_rate,
        dropout=0.1,
        time2vec_k=None,
    ):
        super().__init__(learning_rate)
        self.use_time2vec = time2vec_k is not None
        self.model = GRURegressor(
            num_activities=num_activities,
            hidden_size=hidden_size,
            num_gru_layers=num_layers,
            time2vec_k=time2vec_k,
            dropout=dropout, 
        )
    
    def forward(self, batch):
        if not self.use_time2vec:
            return self.model(batch[ACTIVITIES_SEQUENCE][:-1])
        else:
            return self.model(batch[ACTIVITIES_SEQUENCE][:-1], batch[TIME_SEQUENCE][:-1])


class GRUForBOP(base_tasks.BaseBOP):
    def __init__(
        self,
        num_activities,
        hidden_size,
        num_layers,
        learning_rate: float,
        bidirectional=False,
        time2vec_k=None,
        dropout=0.1,
    ):
        super().__init__(learning_rate)
        self.use_time2vec = time2vec_k is not None
        self.model = GRUClassifier(
            num_activities=num_activities,
            num_classes=2,
            hidden_size=hidden_size,
            num_gru_layers=num_layers,
            bidirectional=bidirectional,
            time2vec_k=time2vec_k,
            dropout=dropout,
        )
    
    def forward(self, batch):
        if not self.use_time2vec:
            return self.model(batch[ACTIVITIES_SEQUENCE])
        else:
            return self.model(batch[TIME_SEQUENCE])
