import dataclasses
from functools import lru_cache
import json
import typing as tp
from argparse import ArgumentParser
from pprint import pprint

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
from data import event_log_data_module
from data.datasets import base_event_loader
from data.datasets.dataset_to_module import DATASET_TO_MODULE
from models.slug_to_class import SLUG_TO_CLASS as MODEL_SLUG_TO_CLASS
from preprocessing import preprocessors as preprocessors_module
from preprocessing.preprocessor_to_module import PREPROCESSOR_SLUG_TO_CLASS
from utils.colors import ColorPrint as colors
from utils.wrappers import JobWithMessage
from constants import *


def _build_preprocessor(
    class_obj,
    **kwargs,
) -> preprocessors_module.BasePreprocessor:
    var_names = class_obj.__init__.__code__.co_varnames
    print(var_names)
    return class_obj(
        **{
            var_name: kwargs.get(var_name)
            for var_name in var_names
            if var_name != "self"
        }
    )

def build_preprocessors(
    preprocessor_slugs: tp.List[str],
    dataset_name,
    event_log,
    graph_builder_config,
    time_type,
) -> tp.List[preprocessors_module.BasePreprocessor]:
    activity2id = preprocessors_module.calculate_activity2id(event_log)
    target_extractor_classes = DATASET_TO_MODULE[dataset_name].get("target_extractors", [])
    target_extractor_classes = [
        _build_preprocessor(class_obj, activity2id=activity2id, relatime=(time_type == 'relative'))
        for class_obj in target_extractor_classes
    ] 

    preprocessor_classes = [
        _build_preprocessor(
            PREPROCESSOR_SLUG_TO_CLASS[slug],
            slug=slug,
            activity2id=activity2id,
            event_log=event_log,
            **graph_builder_config,
        )
        for slug in preprocessor_slugs
    ]

    return tuple([
        *preprocessor_classes,
        *target_extractor_classes,
    ])


@lru_cache
def build_data_module(
    dataset: str,
    train_events,
    val_events,
    test_events,
    batch_size: int,
    num_activities: int,
    preprocessors: tp.Optional[tp.Tuple[preprocessors_module.BasePreprocessor]],
):

    return event_log_data_module.EventLogDataModule(
        train_event_log=train_events,
        val_event_log=val_events,
        test_event_log=test_events,
        batch_size=batch_size,
        num_activities=num_activities,
        preprocessors=preprocessors,
    )

@lru_cache
def load_events(
    dataset_name: str,
    path_to_train: str,
    path_to_val: str,
    path_to_test: str,
    subset: str,
):
    loader_class = DATASET_TO_MODULE[dataset_name]["loader"]
    event_loader: base_event_loader.BaseEventLoader = loader_class()
    train_events = event_loader(path_to_train, subset)
    val_events = event_loader(path_to_val, subset)
    test_events = event_loader(path_to_test, subset)
    return train_events, val_events, test_events



def build_model(
    model: str,
    task: str,
    model_config: tp.Dict[str, tp.Any],
):
    model_slug = model + "_" + task
    model_class = MODEL_SLUG_TO_CLASS[model_slug]
    model = model_class.from_configuration(model_config)
    return model


@dataclasses.dataclass
class Config:
    dataset: str
    model: str
    task: str
    group: str
    batch_size: int
    preprocessor_slugs: tp.List[str]
    model_config: tp.Dict[str, tp.Any]
    train_config: tp.Dict[str, tp.Any]
    graph_builder_config: tp.Dict[str, tp.Any]
    run_name: str
    embeddings: tp.Dict[str, tp.Any]
    process_subset: str
    time_type: str

    def to_dict(self):
        return {field: getattr(self, field) for field in self.__annotations__}

    @classmethod
    def from_config(cls, path_to_config: str):
        with open(path_to_config, "r") as fin:
            config = json.load(fin)
            return cls(
                dataset=config["dataset"],
                batch_size=config["batch_size"],
                preprocessor_slugs=config["preprocessor_slugs"],
                model_config=config["model_config"],
                train_config=config["train_config"],
                run_name=config["run_name"],
                embeddings=config["embeddings"],
                model=config["model"],
                task=config["task"],
                group=config["group"],
                graph_builder_config=config.get("graph_builder_config", {}),
                process_subset=config.get("process_subset"),
                time_type=config['time_type'],
            )

def extract_raw_dataset_paths(config):
    train = f'./raw_data/{config.dataset}/train.csv'
    val = f'./raw_data/{config.dataset}/val.csv'
    test = f'./raw_data/{config.dataset}/test.csv'
    return train, val, test
    

def run_pipeline(config, data_module=None):
    with JobWithMessage("Loading wandb..."):
        wandb.init(
            project='gnn4pbpm',
            name=config.run_name,
            group=config.group,
        )
        logger = pl.loggers.WandbLogger()
    path_to_train, path_to_val, path_to_test = extract_raw_dataset_paths(config)

    if data_module is None: 
        with JobWithMessage("Building data..."):
            train_events, val_events, test_events = load_events(
                config.dataset,
                path_to_train,
                path_to_val,
                path_to_test,
                config.process_subset
            )

            activity2id = preprocessors_module.calculate_activity2id(train_events)
            config.model_config['num_activities'] = len(activity2id)
            logger.experiment.config.update(config.to_dict())
            preprocessors = build_preprocessors(
                dataset_name=config.dataset,
                preprocessor_slugs=config.preprocessor_slugs,
                event_log=train_events,
                graph_builder_config=config.graph_builder_config,
                time_type=config.time_type,
            )
            data_module = build_data_module(
                dataset=config.dataset,
                train_events=train_events,
                val_events=val_events,
                test_events=test_events,
                batch_size=config.batch_size,
                preprocessors=preprocessors,
                num_activities=len(activity2id),
            )
    else:
        print("KEKE")
        config.model_config['num_activities'] = data_module.num_activities
        logger.experiment.config.update(config.to_dict())
        
        
    
    with JobWithMessage("Setup model..."):
        model = build_model(
            model=config.model,
            task=config.task,
            model_config=config.model_config,
        )
        logger.watch(model)
        trainer = pl.Trainer(
            logger=logger,
            **config.train_config, 
            log_every_n_steps=10,
            val_check_interval=0.5,
            callbacks=[
                EarlyStopping(
                    monitor=LOG_LOSS_VAL,
                    mode="min",
                    patience=6,
                ),
                ModelCheckpoint(monitor=LOG_LOSS_VAL)
            ]
        )

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    trainer.fit(model, train_dataloader, val_dataloader)

    test_dataloader = data_module.test_dataloader()

    trainer.test(model, dataloaders=test_dataloader, ckpt_path='best')
    return data_module


def main():
    parser = ArgumentParser()
    parser.add_argument("--path_to_config", default=None)
    args = parser.parse_args()

    with JobWithMessage("Loading config..."):
        config = Config.from_config(args.path_to_config)

    print(colors.format("Here is the config:", colors.WARNING))
    pprint(config.to_dict())
    run_pipeline(config)


if __name__ == "__main__":
    main()

