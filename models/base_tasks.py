from torch import nn
import torch

import typing as tp
import torchmetrics

import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
import torch
from constants import *

from models.architectures.gru import *
from abc import ABC
from abc import abstractmethod


class BaseNAP(pl.LightningModule, ABC):
    def __init__(
        self,
        learning_rate: float
    ):
        super().__init__()
        self.model = nn.Identity()
        self.learning_rate = learning_rate
    
    @classmethod
    def from_configuration(cls, config: tp.Dict[str, tp.Any]):
        return cls(**config)

    @abstractmethod    
    def forward(self, batch):
        raise NotImplementedError

    def _calc_loss_and_acc(self, batch):
        logits = self(batch)
        true_activities = batch[ACTIVITIES_SEQUENCE][1:]
        padding_mask = batch[PADDING_MASK][1:]
        loss = F.cross_entropy(
            logits.permute(1, 2, 0),
            true_activities.permute(1, 0),
            reduction="none",
        ).T
        loss = loss.masked_select(padding_mask).mean()
        preds = logits.argmax(dim=-1)
        hits = ((preds == true_activities) * padding_mask).sum()
        acc =  hits / padding_mask.sum()
        return {LOSS: loss, ACC: acc.item(), BSIZE: padding_mask.sum().item()}


    def training_step(self, batch, batch_i):
        """
            batch[ACTIVITIES_SEQUENCE] has shape [max_len, batch_size]
            batch[PADDING_MASK] has shape[max_len, batch_size]
        """
        output = self._calc_loss_and_acc(batch)
        self.log(LOG_ACC_TRAIN, output[ACC])
        self.log(LOG_LOSS_TRAIN, output[LOSS])
        return output

    def validation_step(self, batch, batch_i):
        output = self._calc_loss_and_acc(batch)
        self.log(LOG_ACC_VAL, output[ACC], batch_size=output[BSIZE], prog_bar=True)
        self.log(LOG_LOSS_VAL, output[LOSS], batch_size=output[BSIZE], prog_bar=True)
        return output

    def test_step(self, batch, i):
        output = self._calc_loss_and_acc(batch)
        self.log(LOG_ACC_TEST, output[ACC], batch_size=output[BSIZE], prog_bar=True)
        self.log(LOG_LOSS_TEST, output[LOSS], batch_size=output[BSIZE], prog_bar=True)
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class BaseNTP(pl.LightningModule, ABC):
    def __init__(
        self, learning_rate: float
    ):
        super().__init__()
        self.model = nn.Identity() 
        self.learning_rate = learning_rate
    
    @classmethod
    def from_configuration(cls, config: tp.Dict[str, tp.Any]):
        return cls(**config)

    @abstractmethod 
    def forward(self, batch):
        raise NotImplementedError

    def _calc_loss_and_mae(self, batch):
        logits = self(batch)
        true_timestamps = batch[NTP_TARGET]
        padding_mask = batch[PADDING_MASK][1:]
        loss = F.mse_loss(
            torch.log1p(logits.T),
            torch.log1p(true_timestamps), 
            reduction="none",
        ).T
        loss = (loss * padding_mask).sum() / padding_mask.sum()
        mae = (torch.abs(logits - true_timestamps.T) * padding_mask).sum() / padding_mask.sum()
        return {LOSS: loss, MAE: mae.item(), BSIZE: padding_mask.sum().item()}

    def training_step(self, batch, batch_i):
        output = self._calc_loss_and_mae(batch)
        self.log(LOG_LOSS_TRAIN, output[LOSS])
        self.log(LOG_MAE_TRAIN, output[MAE])
        return output

    def validation_step(self, batch, batch_i):
        output = self._calc_loss_and_mae(batch)
        self.log(LOG_MAE_VAL, output[MAE], prog_bar=True, batch_size=output[BSIZE])
        self.log(LOG_LOSS_VAL, output[LOSS], prog_bar=True, batch_size=output[BSIZE])
        return output

    def test_step(self, batch, i):
        output = self._calc_loss_and_mae(batch)
        self.log(LOG_MAE_TEST, output[MAE], prog_bar=True, batch_size=output[BSIZE])
        self.log(LOG_LOSS_TEST, output[LOSS], prog_bar=True, batch_size=output[BSIZE])
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class BaseBOP(pl.LightningModule, ABC):
    def __init__(
        self, learning_rate: float
    ):
        super().__init__()
        self.model = nn.Identity()
        self.learning_rate = learning_rate
        self.accuracy = torchmetrics.Accuracy()
        self.auc_roc = torchmetrics.AUROC()
        self.val_accuracy = self.accuracy.clone()
    
    @classmethod
    def from_configuration(cls, config: tp.Dict[str, tp.Any]):
        return cls(**config)

    @abstractmethod    
    def forward(self, batch):
        raise NotImplementedError

    def calc_loss_and_metrics(self, batch):
        logits = self(batch)
        target = batch[BOP_TARGET]
        loss = F.cross_entropy(logits, target)
        preds = logits.argmax(dim=1)
        accuracy = self.accuracy(preds, target)
        auc_roc = self.auc_roc(logits[:, 1], target)
        return {LOSS: loss, "acc": accuracy, "roc_auc": auc_roc, BSIZE: len(target)}

    def training_step(self, batch, batch_i):
        """
            batch[ACTIVITIES_SEQUENCE] has shape [max_len, batch_size]
            batch[PADDING_MASK] has shape[max_len, batch_size]
        """
        output = self.calc_loss_and_metrics(batch)
        self.log(LOG_ACC_TRAIN, self.accuracy, batch_size=output[BSIZE])
        self.log(LOG_LOSS_TRAIN, output[LOSS], batch_size=output[BSIZE])
        self.log(LOG_AUC_TRAIN, self.auc_roc, batch_size=output[BSIZE])
        return output

    def validation_step(self, batch: tp.Dict[str, tp.Any], batch_idx: int):
        output = self.calc_loss_and_metrics(batch)
        self.log(LOG_ACC_VAL, self.accuracy)
        self.log(LOG_LOSS_VAL, output[LOSS])
        self.log(LOG_AUC_VAL, self.auc_roc)

    def test_step(self, batch: tp.Dict[str, tp.Any], batch_idx: int):
        output = self.calc_loss_and_metrics(batch)
        self.log(LOG_ACC_VAL, self.accuracy)
        self.log(LOG_LOSS_TEST, output[LOSS])
        self.log(LOG_AUC_TEST, self.auc_roc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
