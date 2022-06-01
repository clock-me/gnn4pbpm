
import torch
from data import event_log_data_module
from tests import test_utils

def test_target_collate():
    collater = event_log_data_module.UniversalCollater()
    batch = [
        {
            NAP_TARGET: [3, 4, 5],
            NTP_TARGET: [0.0, 2.0, 3.0],
            BOP_TARGET: 0,
        },
        {
            NAP_TARGET: [3, 4, 5, 6],
            NTP_TARGET: [0.0, 2.0, 3.0, 4.0],
            BOP_TARGET: 1,
        }
    ] 
    collated = collater(batch)
    assert torch.all(collated[NAP_TARGET] == torch.tensor([
        [3, 4, 5, 0],
        [3, 4, 5, 6],
    ]))
    assert torch.allclose(collated[NTP_TARGET], torch.tensor([
        [0.0, 2.0, 3.0, 0],
        [0.0, 2.0, 3.0, 4.0],
    ]))
    assert torch.all(collated[BOP_TARGET] == torch.tensor([0, 1]))

def test_activities_sequence():
    collater = event_log_data_module.UniversalCollater()
    batch = [
        {
            ACTIVITIES_SEQUENCE: [3, 4, 5],
        },
        {
            ACTIVITIES_SEQUENCE: [3, 5, 6, 7]
        }
    ] 
    collated = collater(batch)
    assert torch.all(collated[ACTIVITIES_SEQUENCE] == torch.tensor([
        [3, 4, 5, 0],
        [3, 5, 6, 7],
    ]).T)
    assert torch.all(collated[PADDING_MASK] == torch.tensor([
        [True, True, True, False],
        [True, True, True, True]
    ]).T)
    

def test_graph():
    collater = event_log_data_module.UniversalCollater()
    batch = [
        {
            SEQUENTIAL_GRAPH: test_utils.make_sequential_graph(2),
        },
        {
            SEQUENTIAL_GRAPH: test_utils.make_sequential_graph(3),
        }
    ] 
    collated = collater(batch)
    res = collated[SEQUENTIAL_GRAPH]
    assert len(res) == 3

    assert all(
        torch.all(res[i].x == torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]))
        for i in range(len(res))
    )

    assert torch.all(res[0].edge_index == torch.tensor([
        [0, 1],
        [4, 5],
    ]).T)
    assert torch.all(res[1].edge_index == torch.tensor([
        [0, 1],
        [1, 2],
        [4, 5],
        [5, 6],
    ]).T)
    assert torch.all(res[2].edge_index == torch.tensor([
        [0, 1],
        [1, 2],
        [4, 5],
        [5, 6],
        [6, 7],
    ]).T)
