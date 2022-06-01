import typing as tp

from pm4py.objects.log import obj
from sympy import O

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


class NextActivityTargetExtractor(preprocessors.BasePreprocessor):
    def __init__(self, activity2id: tp.Dict[str, int]):
        super().__init__(key="nap_target")
        self.activity2id = activity2id

    def __call__(self, data_item: tp.Dict[str, tp.Any]) -> int:
        return [self.activity2id[event["concept:name"]] for event in data_item["trace"]]


def _extract_difference(lhs_dttm, rhs_dttm):
    seconds = (rhs_dttm - lhs_dttm).seconds
    return seconds

class NextTimestampTargetExtractor(preprocessors.BasePreprocessor):
    def __init__(self):
        super().__init__(key=NTP_TARGET)

    def __call__(self, data_item: tp.Dict[str, tp.Any]) -> int:
        """
        outputs list[float] of size len(trace)
        """
        trace = data_item["trace"]
        result = [0.0]
        for i in range(1, len(trace)):
            result.append(_extract_difference(trace[i][TIMESTAMP_KEY], trace[i - 1][TIMESTAMP_KEY]))
        return result