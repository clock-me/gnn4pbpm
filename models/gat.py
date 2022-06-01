from constants import *

from models.architectures import gat
from models import base_tasks
from time import time

class GATForNTP(base_tasks.BaseNTP):
    def __init__(
        self,
        num_node_types: int,
        num_layers: int,
        num_heads: int,
        hidden_size: int,
        learning_rate: float,
        add_freq_features: bool,
        add_time_features: bool,
        add_type_features: bool,
    ):
        super().__init__(learning_rate)
        self.model = gat.GATRegressor(
            num_activities=num_node_types,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=hidden_size,
            add_freq_features=add_freq_features,
            add_time_features=add_time_features,
            add_type_features=add_type_features,
        )

    def forward(self, batch):
        return self.model(batch[SEQUENTIAL_GRAPH])


class GATForNAP(base_tasks.BaseNAP):
    def __init__(
        self,
        num_activities: int,
        num_node_types: int,
        num_layers: int,
        num_heads: int,
        hidden_size: int,
        learning_rate: float,
        add_freq_features: bool,
        add_time_features: bool,
        add_type_features: bool,
    ):
        super().__init__(learning_rate)
        self.model = gat.GATByTokenClassifier(
            num_activities=num_node_types,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=hidden_size,
            num_classes=num_activities,
            add_freq_features=add_freq_features,
            add_time_features=add_time_features,
            add_type_features=add_type_features,
        )

    def forward(self, batch):
        start = time()
        res = self.model(batch[SEQUENTIAL_GRAPH], batch['seq_graph_lengths'])
        return res

class GATForBOP(base_tasks.BaseBOP):
    def __init__(
        self,
        num_node_types: int,
        num_layers: int,
        num_heads: int,
        hidden_size: int,
        learning_rate: float,
        add_freq_features: bool,
        add_time_features: bool,
        add_type_features: bool,
    ):
        super().__init__(learning_rate)
        self.model = gat.GATClassifier(
            num_activities=num_node_types,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=hidden_size,
            num_classes=2,
            add_freq_features=add_freq_features,
            add_time_features=add_time_features,
            add_type_features=add_type_features,
        )

    def forward(self, batch):
        return self.model(batch[GRAPH])