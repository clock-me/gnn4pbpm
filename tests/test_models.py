from data.event_log_data_module import UniversalCollater
from models import vanilla_ggnn
from models import vanilla_gru
from models import vanilla_transformer
from models.slug_to_class import SLUG_TO_CLASS
import pytest
import json
import torch
from tests import test_utils
from constants import *

@pytest.fixture()
def model_configs():
    with open("tests/static/model_configs.json", "r") as fin:
        configs = json.load(fin)
    return configs

@pytest.fixture()
def batch():
    batch = [
        {
            SEQUENTIAL_GRAPH: test_utils.make_sequential_graph(2),
            GRAPH: test_utils.make_graph(2),
            ACTIVITIES_SEQUENCE: [1, 3, 4],
            NAP_TARGET: [3, 4],
            NTP_TARGET: [0.0, 2.0],
            BOP_TARGET: 0,
        },
        {
            SEQUENTIAL_GRAPH: test_utils.make_sequential_graph(3),
            GRAPH: test_utils.make_graph(3),
            ACTIVITIES_SEQUENCE: [1, 3, 4, 5],
            NAP_TARGET: [3, 4, 5],
            NTP_TARGET: [0.0, 2.0, 3.0],
            BOP_TARGET: 1,
        }
    ] 
    collator = UniversalCollater()
    return collator(batch)
    

def test_model_train_val_steps(model_configs, batch):
    for model_slug in model_configs:
        model_config = model_configs[model_slug]
        model_class = SLUG_TO_CLASS[model_slug]
        model = model_class.from_configuration(model_config) 
        output_train = model.training_step(batch, 0)
        output_val = model.validation_step(batch, 0)
        assert LOSS in output_train
        if model_slug.endswith("bop"):
            assert LOSS in output_val
        else:
            assert "sum_loss" in output_val