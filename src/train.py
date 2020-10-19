import argparse
import json
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


class TrainSettingsDecoder:
    def decode(self, args: argparse.Namespace, network: nn.Module) -> TrainSettings:
        optimizer = self.decode_optimizer_json(network, args.optimizer_json)
        scheduler = self.decode_scheduler_json(optimizer, args.optimizer_json)
        settings = TrainSettings(
            start_epoch=1,
            end_epoch=args.max_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        return settings

    def decode_optimizer_json(
        self, network: nn.Module, path: str
    ) -> optimization.OPTIMIZER_TYPE:
        return optimization.OptimizerRepository().get_optimizer(
            network, args.optimizer_json
        )

    def decode_scheduler_json(
        self, optimizer: optimization.OPTIMIZER_TYPE, path: str
    ) -> optimization.SCHEDULER_TYPE:
        return optimization.SchedulerRepository().get_scheduler(
            optimizer, args.optimizer_json
        )


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


def main(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = model.utils.get_model(args).to(device)
    criterion = model.utils.get_criterion(args)
    settings = TrainSettingsDecoder().decode(args, network)
    mean, std = dataset.get_stats()
    spatial_transforms = dataset.SpatialTransformRepository(
        mean, std
    ).get_transform_obj(args.transforms_json)
    temporal_transforms = dataset.TemporalTransformRepository().get_transform_obj(
        args.transforms_json
    )
    with open(args.config_json) as f:
        config: Mapping[str, Any] = json.load(f)
    videodata_repository = dataset.get_dataset(
        args, spatial_transforms, temporal_transforms, **config,
    )
    train_loader = DataLoader(
        videodata_repository,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_data,
        drop_last=False,
    )
    trainer = Trainer(settings, network, criterion, device)
    trainer.train(train_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset essentials
    parser.add_argument(
        "--root-path", type=str, default="/home/aab10821no/datasets/kinetics"
    )
    parser.add_argument("--hdf-path", type=str, default="videos_700_hdf5")
    parser.add_argument("--ann-path", type=str, default="kinetics-700-hdf5.json")
    parser.add_argument("--model", type=str, default="dpc")

    # training config
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)

    # settings
    parser.add_argument("--config-json", type=str, default="default_config.json")
    parser.add_argument(
        "--optimizer-json", type=str, default="default_optimizer.json",
    )
    parser.add_argument(
        "--transforms-json", type=str, default="default_transforms.json"
    )
    args = parser.parse_args()
    main(args)
