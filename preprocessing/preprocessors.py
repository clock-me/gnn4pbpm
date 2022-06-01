import abc
import typing as tp

from pm4py.objects.log import obj
from constants import *
import constants 

def calculate_activity2id(event_log: obj.EventLog):
    unique_activities = set()
    for trace in event_log:
        for event in trace:
            unique_activities.add(event["concept:name"])
    return {
        constants.BOS: constants.BOS_i,
        constants.EOS: constants.EOS_i,
        constants.PAD: constants.PAD_i, 
        **{
            activity: idx + 3
            for idx, activity in enumerate(unique_activities)
        }
    }


class BasePreprocessor(abc.ABC):
    def __init__(
        self,
        key: str,
    ):
        self.key = key

    @abc.abstractmethod
    def __call__(self, data_item: tp.Dict[str, tp.Any]) -> tp.Any:
        raise NotImplementedError
    
    def __hash__(self) -> int:
        return hash(repr(self))


class ActivitySequenceExtractor(BasePreprocessor):
    def __init__(self, activity2id: tp.Dict[str, int]):
        super().__init__(ACTIVITIES_SEQUENCE)
        self.activity2id = activity2id
    
    def __call__(self, data_item: tp.Dict[str, tp.Any]) -> tp.Any:
        return [
            constants.BOS_i,
            *[self.activity2id[trace["concept:name"]] for trace in data_item["trace"]],
        ]

def _extract_difference(lhs_dttm, rhs_dttm):
    seconds = (lhs_dttm - rhs_dttm).total_seconds() / 86400
    return seconds


TIMESTAMP_KEY = "time:timestamp"


class TimeSequenceExtractor(BasePreprocessor):
    def __init__(
        self,
        activity2id: tp.Dict[str, int],
        relative: bool
    ):
        super().__init__(TIME_SEQUENCE)
        self.activity2id = activity2id
        self.relative = relative
    
    def __call__(self, data_item: tp.Dict[str, tp.Any]) -> tp.Any:
        trace = data_item["trace"]
        result = [0.0, 0.0]
        for i in range(1, len(trace)):
            if self.relative:
                result.append(_extract_difference(trace[i][TIMESTAMP_KEY], trace[i - 1][TIMESTAMP_KEY]))
            else:
                result.append(_extract_difference(trace[i][TIMESTAMP_KEY], trace[0][TIMESTAMP_KEY]))
        return result