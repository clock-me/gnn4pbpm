from data.datasets.bpi2017.loader import EventLoader as BPI2017Loader
from data.datasets.helpdesk.loader import EventLoader as HelpdeskLoader
from data.datasets.bpi2012.loader import EventLoader as BPI2012Loader

DATASET_TO_MODULE = {
    "bpi2017": {
        "loader": BPI2017Loader,
    },
    "helpdesk": {
        "loader": HelpdeskLoader,
    },
    "bpi2012": {
        "loader": BPI2012Loader,
    },
}
