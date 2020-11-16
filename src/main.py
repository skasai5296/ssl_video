import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from lib import dataset, model, optimization, train


class TrainSettingsDecoder:
    def decode(
        self, args: argparse.Namespace, network: nn.Module
    ) -> train.TrainSettings:
        optimizer = self.decode_optimizer_json(network, args.optimizer_json)
        scheduler = self.decode_scheduler_json(optimizer, args.optimizer_json)
        settings = train.TrainSettings(
            start_epoch=1,
            end_epoch=args.max_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        return settings

    def decode_optimizer_json(
        self, network: nn.Module, path: str
    ) -> optimization.OPTIMIZER_TYPE:
        return optimization.OptimizerRepository().get_optimizer(network, path)

    def decode_scheduler_json(
        self, optimizer: optimization.OPTIMIZER_TYPE, path: str
    ) -> optimization.SCHEDULER_TYPE:
        return optimization.SchedulerRepository().get_scheduler(optimizer, path)


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
    trainer = train.Trainer(settings, network, criterion, device)
    trainer.train(train_loader)


def make_absolute_path(path: str) -> str:
    curdir = os.path.dirname(__file__)
    return os.path.normpath(os.path.join(curdir, path))


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
    parser.add_argument(
        "--config-json", type=make_absolute_path, default="default_config.json"
    )
    parser.add_argument(
        "--optimizer-json", type=make_absolute_path, default="default_optimizer.json",
    )
    parser.add_argument(
        "--transforms-json", type=make_absolute_path, default="default_transforms.json"
    )
    args = parser.parse_args()
    main(args)
