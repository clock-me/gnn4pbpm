from preprocessing.graph_builders import StaticGraphBuilder
from preprocessing.preprocessors import ActivitySequenceExtractor
from preprocessing.preprocessors import TimeSequenceExtractor

PREPROCESSOR_SLUG_TO_CLASS = {
    "StaticGraphBuilder": StaticGraphBuilder,
    "ActivitySequenceExtractor": ActivitySequenceExtractor,
    "TimeSequenceExtractor": TimeSequenceExtractor,
}
