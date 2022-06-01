from re import U
from unittest import result
from preprocessing import graph_builders, preprocessors
import torch
import constants
from pm4py.objects.log import obj
import pytest

activity2id = {
    constants.BOS: constants.BOS_i,
    constants.EOS: constants.EOS_i,
    constants.PAD: constants.PAD_i,
    "a": 3,
    "b": 4,
    "c": 5,
}

@pytest.mark.parametrize(
    "trace, result_x, result_edge_index",
    [
        (
            obj.Trace([
                {"concept:name": "a"},
                {"concept:name": "a"},
                {"concept:name": "b"},
                {"concept:name": "c"},
                {"concept:name": "c"},
            ]),
            torch.tensor([0, 1, 2, 3, 4, 5]),
            torch.tensor([
                [constants.BOS_i, 3],
                [3, 3],
                [3, 4],
                [4, 5],
                [5, 5],
            ]).T 
        ),
        (
            obj.Trace([
            ]),
            torch.tensor([0, 1, 2, 3, 4, 5]),
            None,
        ),
        (
            obj.Trace([
                {"concept:name": "a"},
            ]),
            torch.tensor([0, 1, 2, 3, 4, 5]),
            torch.tensor([
                [constants.BOS_i, 3],
            ]).T 
        ),
    ]
)
def test_dfg_graph_builder(trace, result_x, result_edge_index):
    graph_builder = graph_builders.StaticGraphBuilder()
    data_item = {
        "trace": trace,
    }
    graph = graph_builder(data_item)
    assert torch.all(graph.x == result_x)
    if result_edge_index is None:
        assert graph.edge_index is None
    else:
        assert torch.all(graph.edge_index == result_edge_index).T
    

def test_sequential_prefix_graph_builder():
    trace = obj.Trace([
        {"concept:name": "a"},
        {"concept:name": "a"},
        {"concept:name": "b"},
        {"concept:name": "c"},
        {"concept:name": "c"},
    ]),
    sequential_graph_builder = (
        graph_builders.SequentialDFGGraphBuilder(activity2id)
    )
    result = sequential_graph_builder({
        "trace": trace
    })
    assert len(result) == len(trace)