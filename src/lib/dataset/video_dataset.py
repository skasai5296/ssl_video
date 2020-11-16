import enum
import glob
import io
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import h5py
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import metadata, spatial_transforms, temporal_transforms


class VideoLoaderHDF5:
    def __call__(self, video_path: str, frame_indices: List[int]) -> List[Image.Image]:
        with h5py.File(video_path, "r") as f:
            video_data = f["video"]

            video = []
            for i in frame_indices:
                if i < len(video_data):
                    video.append(Image.open(io.BytesIO(video_data[i])))
                else:
                    return video

        return video


@dataclass
class ClipData:
    label: int
    clip: torch.Tensor


@dataclass
class ClipBatch:
    label: torch.Tensor
    clip: torch.Tensor


def collate_data(data: List[ClipData]) -> ClipBatch:
    labels = []
    clips = []
    for d in data:
        labels.append(d.label)
        clips.append(d.clip)
    return ClipBatch(
        label=torch.tensor(labels, dtype=torch.long), clip=torch.stack(clips),
    )


class VideodataRepository(Dataset):
    def __init__(
        self,
        metadata: List[metadata.VideoMetadata],
        clip_len: int,
        n_clip: int,
        downsample: int,
        spatial_transform: spatial_transforms.Compose,
        temporal_transform: temporal_transforms.Compose,
        mode: metadata.Mode = metadata.Mode.TRAIN,
    ):
        self.loader = VideoLoaderHDF5()
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.clip_len = clip_len
        self.n_clip = n_clip
        self.downsample = downsample
        self.mode = mode
        # minimum_clip_frames = clip_len * n_clip * downsample

        self.metadata = metadata
        self.classes: List[str] = list(set(meta.label for meta in metadata))

    def clipify(self, tensor: torch.Tensor, clip_len: int) -> torch.Tensor:
        """
        Divide tensor of video frames into clips
        Args:
            tensor: torch.Tensor(C, n_clip*clip_len, H, W)
            clip_len: int, number of frames for a single clip
        Returns:
            torch.Tensor(n_clip, C, clip_len, H, W), sampled clips
        """
        assert [*tensor.size()][:2] == [3, self.n_clip * self.clip_len]
        assert tensor.dim() == 4
        split = torch.split(tensor, clip_len, dim=1)
        stacked = torch.stack(split)
        assert [*stacked.size()][:3] == [self.n_clip, 3, self.clip_len]
        assert stacked.dim() == 5
        return stacked

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> ClipData:
        """
        Get tensor of video.
        Returns:
            Videodata containing the clip (n_clip, C, clip_len, H, W)
        """
        obj = self.metadata[index]
        start, end = obj.duration
        frame_indices = list(range(start - 1, end - 1))
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
            assert len(frame_indices) == self.clip_len * self.n_clip
        clip = self.loader(obj.path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip_tensor = torch.stack([self.spatial_transform(img) for img in clip], 1)
        clip_tensor = self.clipify(clip_tensor, self.clip_len)
        label = self.classes.index(obj.label)
        return ClipData(label=label, clip=clip_tensor)
