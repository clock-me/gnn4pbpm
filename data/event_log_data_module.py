import typing as tp
from time import time

import torch
import torch.utils.data as torch_data_utils
import pytorch_lightning as pl
import torch_geometric.data as tg_data_utils
from pm4py.objects.log import obj

from data.datasets import dataset
from embeddings.base import BaseEmbedding
from preprocessing import preprocessors as preprocessors_module
from copy import deepcopy
from constants import *


class UniversalCollater:
    def __init__(self, *args, **argv) -> None:
        pass

    def __call__(self, batch):
        # batch = deepcopy(batch_)
        result = {}

        start = time()
        if SEQUENTIAL_GRAPH in batch[0]:
            sequential_graphs = [b[SEQUENTIAL_GRAPH] for b in batch]
            sequential_graph_lengths = [len(b[SEQUENTIAL_GRAPH]) for b in batch]
            all_batched = tg_data_utils.Batch.from_data_list([
                g for seq in sequential_graphs for g in seq
            ])
            result['seq_graph_lengths'] = torch.tensor(sequential_graph_lengths)
            result[SEQUENTIAL_GRAPH] = all_batched
        end = time()
        if GRAPH in batch[0]:
            graphs = [b[GRAPH] for b in batch]
            result[GRAPH] = tg_data_utils.Batch.from_data_list(graphs)

        if BOP_TARGET in batch[0]:
            bop_target = [b[BOP_TARGET] for b in batch]
            result[BOP_TARGET] = torch.tensor(bop_target)
        
        lenghts = None

        if NAP_TARGET in batch[0]:
            nap_targets = [b[NAP_TARGET] for b in batch]
            lenghts = [len(g) for g in nap_targets]
            max_len = max(len(g) for g in nap_targets)
            for i in range(len(nap_targets)):
                while (len(nap_targets[i]) < max_len):
                    nap_targets[i].append(0)

            result[NAP_TARGET] = torch.tensor(nap_targets)

        if NTP_TARGET in batch[0]:
            ntp_targets = [b[NTP_TARGET] for b in batch]
            lenghts_ntp = [len(g) for g in ntp_targets] 
            max_len = max(len(g) for g in ntp_targets)
            if lenghts is not None:
                assert lenghts == lenghts_ntp
            else:
                lenghts = lenghts_ntp
            for i in range(len(ntp_targets)):
                while (len(ntp_targets[i]) < max_len):
                    ntp_targets[i].append(0)

            result[NTP_TARGET] = torch.tensor(ntp_targets)
        if lenghts is not None:
            result[LOSS_MASK] = torch.arange(max(lenghts))[None, :] < torch.tensor(lenghts)[:, None]
        if ACTIVITIES_SEQUENCE in batch[0]:
            activities_sequences = [b[ACTIVITIES_SEQUENCE] for b in batch]
            lenghts = [len(g) for g in activities_sequences]
            max_len = max(lenghts)
            for i in range(len(activities_sequences)):
                while (len(activities_sequences[i]) < max_len):
                    activities_sequences[i].append(0)
            result[PADDING_MASK] = (torch.arange(max(lenghts))[None, :] < torch.tensor(lenghts)[:, None]).T
            result[ACTIVITIES_SEQUENCE] = torch.tensor(activities_sequences).T

        if TIME_SEQUENCE in batch[0]:
            time_sequences = [b[TIME_SEQUENCE] for b in batch]
            lenghts = [len(g) for g in time_sequences]
            max_len = max(lenghts)
            for i in range(len(time_sequences)):
                while (len(time_sequences[i]) < max_len):
                    time_sequences[i].append(0)
            result[TIME_SEQUENCE] = torch.tensor(time_sequences).T
        
        # print(f"collating took {end - start} seconds")
        return result



class EventLogDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_event_log: obj.EventLog,
        val_event_log: obj.EventLog,
        test_event_log: obj.EventLog,
        batch_size: int,
        num_activities: int,
        preprocessors: tp.Optional[tp.List[preprocessors_module.BasePreprocessor]],
    ):
        super().__init__()
        self.event_logs: tp.Dict[str, obj.EventLog] = {
            "train": train_event_log,
            "val": val_event_log,
            "test": test_event_log,
        }
        self.preprocessors = preprocessors
        self.num_activities = num_activities
        self.batch_size = batch_size
        self.datasets: tp.Dict[str, dataset.Dataset] = {
            key: dataset.Dataset(value) for key, value in self.event_logs.items()
        }
        for key in self.datasets:
            self.datasets[key].preprocess(self.preprocessors)

    def download_data(self) -> None:
        raise NotImplementedError

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return torch_data_utils.DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            collate_fn=UniversalCollater(),
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch_data_utils.DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            collate_fn=UniversalCollater(),
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch_data_utils.DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            collate_fn=UniversalCollater(),
            shuffle=False,
            pin_memory=True,
        )

