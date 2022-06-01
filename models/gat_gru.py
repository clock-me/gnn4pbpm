from torch import nn

import torchmetrics

import pytorch_lightning as pl
import torch
from constants import *

from models.architectures.gat_gru import *
from models import base_tasks

class GATGRUForNAP(base_tasks.BaseNAP):
    def __init__(
        self,
        learning_rate,
        num_activities,
        hidden_size,
        num_gru_layers, 
        num_node_types,
        num_gat_heads,
        num_gat_layers,
    ):
        super().__init__(learning_rate)
        self.model = GATGRUByTokenClassifier(
            num_activities,
            hidden_size,
            num_gru_layers,
            num_node_types,
            num_gat_heads,
            num_gat_layers,
            num_activities
        )
    
    def forward(self, batch):
        logits = self.model.forward(
            batch[SEQUENTIAL_GRAPH],
            batch[SEQ_GRAPH_LENGTHS],
            batch[ACTIVITIES_SEQUENCE][:-1],
            batch[TIME_SEQUENCE][:-1]
        )
        return logits
