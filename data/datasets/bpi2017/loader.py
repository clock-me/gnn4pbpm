import pandas as pd
import pm4py
import pm4py.objects.log.obj as data_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils

from .. import base_event_loader


class EventLoader(base_event_loader.BaseEventLoader):
    def __call__(self, path_to_file: str, subset: str) -> data_utils.EventLog:
        df = pd.read_csv(path_to_file, sep=",")
        event_log = self.filter_and_cast_to_pm4py_format(df, subset)
        return event_log

    @staticmethod
    def filter_and_cast_to_pm4py_format(df: pd.DataFrame, subset: str) -> data_utils.EventLog:
        log_csv = dataframe_utils.convert_timestamp_columns_in_df(df)
        if subset is not None: 
            log_csv = log_csv[log_csv['concept:name'].apply(lambda x: x.startswith(subset))]
        log_csv = log_csv.sort_values("time:timestamp")
        event_log = log_converter.apply(
            log_csv,
            parameters={
                log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "case:concept:name"
            },
        )
        event_log = pm4py.filter_log(lambda trace: len(trace) > 2, event_log)
        return event_log
