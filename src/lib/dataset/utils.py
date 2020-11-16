import argparse
import os
from typing import List, Tuple

from lib.dataset import metadata, spatial_transforms, temporal_transforms, video_dataset


def get_stats() -> Tuple[List[float], List[float]]:
    """get mean and std for Kinetics"""
    maximum = 255.0
    mean = [110.63666788 / maximum, 103.16065604 / maximum, 96.29023126 / maximum]
    std = [38.7568578 / maximum, 37.88248729 / maximum, 40.02898126 / maximum]
    return mean, std


def get_dataset(
    args: argparse.Namespace,
    spatial_transform: spatial_transforms.Compose,
    temporal_transform: temporal_transforms.Compose,
    **kwargs,
) -> video_dataset.VideodataRepository:
    if args.model == "dpc":
        clip_len: int = kwargs["clip_len"]
        n_clip: int = kwargs["n_clip"]
        downsample: int = kwargs["downsample"]
        ann_path = os.path.join(args.root_path, args.ann_path)
        hdf_path = os.path.join(args.root_path, args.hdf_path)
        __metadata = metadata.MetadataRepository(hdf_path, metadata.Mode.TRAIN).get_all(
            ann_path
        )
        print(f"# metadata: {len(__metadata)}")
        dataset = video_dataset.VideodataRepository(
            metadata=__metadata,
            clip_len=clip_len,
            n_clip=n_clip,
            downsample=downsample,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
        )
        print(f"# videos: {len(dataset)}")
    else:
        raise NotImplementedError(f"dataset {args.dataset} not supported.")
    return dataset
