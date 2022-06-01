from sympy import O
from preprocessing import preprocessors
from preprocessing import preprocessors as preprocessors_module
from data.datasets.dataset_to_module import DATASET_TO_MODULE
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


def test_activity_sequence_extractor():
    trace = obj.Trace([
        {"concept:name": "a"},
        {"concept:name": "a"},
        {"concept:name": "b"},
        {"concept:name": "c"},
        {"concept:name": "c"},
    ])
    sequence_extractor = preprocessors.ActivitySequenceExtractor(activity2id)
    result = sequence_extractor({"trace": trace})
    assert result == [1, 3, 3, 4, 5, 5]


def _build_preprocessor(
    class_obj,
    **kwargs,
) -> preprocessors_module.BasePreprocessor:
    var_names = class_obj.__init__.__code__.co_varnames
    return class_obj(
        **{
            var_name: kwargs[var_name]
            for var_name in var_names
            if var_name != "self"
        }
    )

@pytest.mark.parametrize(
    "dataset_slug",
    [
        ("bpi2017w")
    ]
)
def test_extractors(dataset_slug):
    target_extractor_classes = DATASET_TO_MODULE[dataset_slug]["target_extractors"]
    assert len(target_extractor_classes)
    for tg_extractor_class in target_extractor_classes:
        target_extractor = _build_preprocessor(tg_extractor_class, activity2id=activity2id)
