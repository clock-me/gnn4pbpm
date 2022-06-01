from preprocessing.process_graph import ProcessGraph
from preprocessing.preprocessors import calculate_activity2id
from preprocessing.graph_builders import StaticGraphBuilder
from preprocessing.preprocessors import TimeSequenceExtractor
from preprocessing.preprocessors import ActivitySequenceExtractor
from constants import *

    
import pandas as pd
import string
from pipeline import load_events
import pm4py
    


def main():
    dataset = 'bpi2017'
    train_events, val_events, test_events = load_events(
        dataset,
        path_to_train=f'/Users/nkpalchikov/diploma/gnn4pbpm/raw_data/{dataset}/train.csv',
        path_to_val=f'/Users/nkpalchikov/diploma/gnn4pbpm/raw_data/{dataset}/val.csv',
        path_to_test=f'/Users/nkpalchikov/diploma/gnn4pbpm/raw_data/{dataset}/test.csv',
        subset='W'
    )
    activity2id = calculate_activity2id(train_events)
    time_extractor = TimeSequenceExtractor(activity2id, relative=True)
    activity_sequence_extractor = ActivitySequenceExtractor(activity2id)
    def make_data_item(trace):
        dt = {
            'trace': trace
        }
        dt[TIME_SEQUENCE] = time_extractor(dt)
        dt[ACTIVITIES_SEQUENCE] = activity_sequence_extractor(dt)
        return dt
    data_item = make_data_item(train_events[0])
    assert len(data_item[TIME_SEQUENCE]) == len(data_item[ACTIVITIES_SEQUENCE])
    for i in range(len(data_item[TIME_SEQUENCE])):
        print(data_item[TIME_SEQUENCE][i], data_item[ACTIVITIES_SEQUENCE][i])
    graph_builder = StaticGraphBuilder(
        activity2id,
        train_events,
        'bpmn',
        add_edges_type='all',
        sequential=False
    )
    graph_builder.visualize_trace(data_item['trace'], data_item[TIME_SEQUENCE], data_item[ACTIVITIES_SEQUENCE])

if __name__ == '__main__':
    main()