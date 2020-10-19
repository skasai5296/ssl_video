import argparse
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from lib.dataset import (SpatialTransformRepository,
                         TemporalTransformRepository, VideoDataRepository,
                         collate_data)
from lib.dataset import utils as dataset_utils
from lib.model import DPC, DPCLoss
from lib.optimization import (OPTIMIZER_TYPE, SCHEDULER_TYPE,
                              OptimizerRepository, SchedulerRepository)


@dataclass
class TrainSettings:
    start_epoch: int
    end_epoch: int
    optimizer: OPTIMIZER_TYPE
    scheduler: SCHEDULER_TYPE


class TrainSettingsDecoder:
    def decode(self, args: argparse.Namespace, model: nn.Module) -> TrainSettings:
        optimizer = OptimizerRepository().get_optimizer(model, args.optimizer_json)
        scheduler = SchedulerRepository().get_scheduler(optimizer, args.optimizer_json)
        settings = TrainSettings(
            start_epoch=1,
            end_epoch=args.max_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        return settings


class Trainer:
    def __init__(
        self,
        settings: TrainSettings,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
    ):
        self.settings = settings
        self.epoch = settings.start_epoch
        self.model = model
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
            self.model.zero_grad()
            out = self.model(data.clip.to(self.device))
            loss, metric = self.criterion(*out)
            loss.backward()
            self.settings.optimizer.step()
            if it % 1 == 0:
                print(f"{it:06d}/{max_it:06d}, {metric}")
        self._scheduler_step(metrics=metric.target_value)

    def train(self, dataloader: DataLoader):
        self.model.train()
        for ep in range(self.settings.start_epoch, self.settings.end_epoch + 1):
            self.train_epoch(dataloader)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DPC(
        args.input_size,
        args.hidden_size,
        args.kernel_size,
        args.num_layers,
        args.n_clip,
        args.pred_step,
        args.dropout,
    ).to(device)
    criterion = DPCLoss()
    train_settings_decoder = TrainSettingsDecoder()
    settings = train_settings_decoder.decode(args, model)
    mean, std = dataset_utils.get_stats()
    spatial_transforms = SpatialTransformRepository(mean, std).get_transform_obj(
        args.transforms_json
    )
    temporal_transforms = TemporalTransformRepository().get_transform_obj(
        args.transforms_json
    )
    videodata_repository = VideoDataRepository(
        args.root_path,
        args.hdf_path,
        args.ann_path,
        args.clip_len,
        args.n_clip,
        args.downsample,
        spatial_transforms,
        temporal_transforms,
    )
    train_loader = DataLoader(
        videodata_repository,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_data,
        drop_last=False,
    )
    trainer = Trainer(settings, model, criterion, device)
    trainer.train(train_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-path", type=str, default="/home/aab10821no/datasets/kinetics"
    )
    parser.add_argument("--hdf-path", type=str, default="videos_700_hdf5")
    parser.add_argument("--ann-path", type=str, default="kinetics-700-hdf5.json")
    parser.add_argument("--clip-len", type=int, default=5)
    parser.add_argument("--n-clip", type=int, default=8)
    parser.add_argument("--downsample", type=int, default=3)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--kernel-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--pred-step", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument(
        "--optimizer-json", type=str, default="default_optimizer.json",
    )
    parser.add_argument(
        "--transforms-json", type=str, default="default_transforms.json"
    )
    args = parser.parse_args()
    main(args)
