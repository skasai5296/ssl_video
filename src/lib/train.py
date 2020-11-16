import os
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from lib import dataset, model, optimization


@dataclass
class TrainSettings:
    start_epoch: int
    end_epoch: int
    optimizer: optimization.OPTIMIZER_TYPE
    scheduler: optimization.SCHEDULER_TYPE


class Trainer:
    def __init__(
        self,
        settings: TrainSettings,
        network: nn.Module,
        criterion: nn.Module,
        device: torch.device,
    ):
        self.settings = settings
        self.epoch = settings.start_epoch
        self.network = network
        self.criterion = criterion
        self.is_plateau = isinstance(
            self.settings.scheduler, lr_scheduler.ReduceLROnPlateau
        )
        self.device = device

    def _scheduler_step(self, metrics: float):
        if self.is_plateau:
            self.settings.scheduler.step(metrics=metrics)
        else:
            self.settings.scheduler.step()

    def train_epoch(self, dataloader: DataLoader):
        max_it: int = len(dataloader)
        for it, data in enumerate(dataloader, 1):
            self.network.zero_grad()
            out = self.network(data.clip.to(self.device))
            loss, metric = self.criterion(*out)
            loss.backward()
            self.settings.optimizer.step()
            if it % 1 == 0:
                print(f"{it:06d}/{max_it:06d}, {metric}")
        self._scheduler_step(metrics=metric.target_value)

    def train(self, dataloader: DataLoader):
        self.network.train()
        for ep in range(self.settings.start_epoch, self.settings.end_epoch + 1):
            self.train_epoch(dataloader)
