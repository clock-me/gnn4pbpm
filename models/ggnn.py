from constants import *

from models.architectures import ggnn
from models import base_tasks


class GGNNForNTP(base_tasks.BaseNTP):
    def __init__(
        self,
        num_activities: int,
        num_timestamps: int,
        hidden_size: int,
        learning_rate: float,
        add_freq_features: bool,
    ):
        super().__init__(learning_rate)
        self.model = ggnn.GGNNRegressor(
            num_activities=num_activities,
            num_timestamps=num_timestamps,
            hidden_size=hidden_size,
            add_freq_features=add_freq_features
        )
    
    def forward(self, batch):
        return self.model(batch[SEQUENTIAL_GRAPH], batch['seq_graph_lengths'])


class GGNNForNAP(base_tasks.BaseNAP):
    def __init__(
        self,
        num_activities: int,
        num_node_types: int,
        num_timestamps: int,
        hidden_size: int,
        learning_rate: float,
        add_freq_features: bool,
    ):
        super().__init__(learning_rate)
        self.model = ggnn.GGNNByTokenClassifier(
            num_activities=num_node_types,
            num_timestamps=num_timestamps,
            hidden_size=hidden_size,
            num_classes=num_activities,
            add_freq_features=add_freq_features,
        )

    def forward(self, batch):
        return self.model(batch[SEQUENTIAL_GRAPH], batch['seq_graph_lengths'])


class GGNNForBOP(base_tasks.BaseBOP):
    def __init__(
        self,
        num_activities: int,
        num_timestamps: int,
        hidden_size: int,
        learning_rate: float,
        add_freq_features: bool
    ):
        super().__init__(learning_rate)
        self.model = ggnn.GGNNClassifier(
            num_activities=num_activities,
            num_timestamps=num_timestamps,
            hidden_size=hidden_size,
            num_classes=2,
        )

    def forward(self, batch):
        return self.model(batch[GRAPH])