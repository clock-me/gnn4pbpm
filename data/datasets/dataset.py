import dataclasses
import typing as tp

import numpy as np
import torch.utils.data as torch_data_utils
from pm4py.objects.log import obj
from torch_geometric import data as tg_data_utils
from tqdm import tqdm

from preprocessing import preprocessors as preprocessor_module
from constants import *
from copy import deepcopy
from time import time


@dataclasses.dataclass
class DatasetItem:
    trace: obj.Trace
    label: int
    graph: tg_data_utils.Data


class Dataset(torch_data_utils.Dataset):
    def __init__(
        self,
        event_log: obj.EventLog,
    ):
        self.event_log = event_log
        self.data = np.array(
            [{"trace_id": trace.attributes["concept:name"], "trace": trace} for trace in self.event_log]
        )

    def preprocess(
        self,
        preprocessors: tp.List[preprocessor_module.BasePreprocessor],
    ):
        for i, trace in tqdm(
            enumerate(self.event_log),
            total=len(self.event_log),
        ):
            for preprocessor in preprocessors:
                self.data[i][preprocessor.key] = preprocessor(
                    self.data[i]
                )

    def augment(self, transforms: list):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tp.Dict[str, tp.Any]:
        return deepcopy(self.data[idx])