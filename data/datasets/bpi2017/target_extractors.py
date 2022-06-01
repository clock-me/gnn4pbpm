import typing as tp

from pm4py.objects.log import obj

from preprocessing import preprocessors
from datetime import datetime
from constants import *


ACTIVITY_KEY = "concept:name"
TIMESTAMP_KEY = "time:timestamp"

class BinaryTargetExtractor(preprocessors.BasePreprocessor):
    def __init__(self):
        super().__init__(key=BOP_TARGET)

    def __call__(self, data_item: tp.Dict[str, tp.Any]) -> int:
        return int(data_item["trace"].attributes["Accepted"])