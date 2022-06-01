from constants import *

from models.architectures import gcn
from models import base_tasks


class GCNForNTP(base_tasks.BaseNTP):
    def __init__(
        self,
        num_activities: int,
        num_layers: int,
        hidden_size: int,
        learning_rate: float,
        add_freq_features: bool,
    ):
        super().__init__(learning_rate)
        self.model = gcn.GCNRE(
            num_activities=num_activities,
            num_layers=num_layers,
            hidden_size=hidden_size,
            add_freq_features=add_freq_features
        )

    def forward(self, batch):
        return self.model(batch[SEQUENTIAL_GRAPH])


class GCNForNAP(base_tasks.BaseNAP):
    def __init__(
        self,
        num_activities: int,
        num_node_types: int,
        num_layers: int,
        hidden_size: int,
        learning_rate: float,
        add_freq_features: bool,
    ):
        super().__init__(learning_rate)
        self.model = gcn.GCNByTokenClassifier(
            num_activities=num_node_types,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_classes=num_activities,
            add_freq_features=add_freq_features,
        )

    def forward(self, batch):
        return self.model(batch[SEQUENTIAL_GRAPH], batch['seq_graph_lengths'])


class GCNForBOP(base_tasks.BaseBOP):
    def __init__(
        self,
        num_activities: int,
        num_layers: int,
        hidden_size: int,
        learning_rate: float,
        add_freq_features: bool
    ):
        super().__init__(learning_rate)
        self.model = gcn.GCNClassifier(
            num_activities=num_activities,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_classes=2,
            add_freq_features=add_freq_features,
        )

    def forward(self, batch):
        return self.model(batch[GRAPH])