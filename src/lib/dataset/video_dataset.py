import enum
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import torch
from torch.utils.data import Dataset

from . import spatial_transforms, temporal_transforms, utils


@dataclass
class VideoMetaData:
    id: str
    path: str
    label: str
    duration: Tuple[int, int]


@dataclass
class VideoData:
    label: int
    clip: torch.Tensor


@dataclass
class VideoBatch:
    label: torch.Tensor
    clip: torch.Tensor


def collate_data(data: List[VideoData]) -> VideoBatch:
    labels = []
    clips = []
    for d in data:
        labels.append(d.label)
        clips.append(d.clip)
    return VideoBatch(
        label=torch.tensor(labels, dtype=torch.long), clip=torch.stack(clips)
    )


class Mode(enum.Enum):
    TRAIN = enum.auto()
    VALIDATION = enum.auto()
    TEST = enum.auto()


class VideoDataRepository(Dataset):
    def __init__(
        self,
        root_path: str,
        hdf_path: str,
        ann_path: str,
        clip_len: int,
        n_clip: int,
        downsample: int,
        spatial_transform: spatial_transforms.Compose,
        temporal_transform: temporal_transforms.Compose,
        mode: Mode = Mode.TRAIN,
    ):
        self.loader = utils.VideoLoaderHDF5()
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.clip_len = clip_len
        self.n_clip = n_clip
        self.downsample = downsample
        self.mode = mode
        minimum_clip_frames = clip_len * n_clip * downsample

        video_path = os.path.join(root_path, hdf_path)
        ann_path = os.path.join(root_path, ann_path)
        with open(ann_path, "r") as f:
            obj = json.load(f)
        labels: Set[str] = set(obj["labels"])
        ann: Dict = obj["database"]
        self.classes: List[str] = []
        self.data: List[VideoMetaData] = []
        failcnt: int = 0
        for classname in self._get_classes_from_root_path(video_path):
            if classname not in self.classes:
                self.classes.append(classname)
            assert classname in labels
            for path in self._get_paths_from_class(video_path, classname):
                video_id = self._get_base_id(path)
                annotation = ann.get(video_id)
                if not annotation:
                    # print(f"id: {video_id} not in annotation")
                    failcnt += 1
                    continue
                split = annotation["subset"]
                if mode == Mode.TRAIN and split != "training":
                    continue
                elif mode == Mode.VALIDATION and split != "validation":
                    continue
                segment: List[int] = annotation["annotations"]["segment"]
                duration: Tuple[int, int] = (segment[0], segment[1])
                if duration[1] < minimum_clip_frames:
                    # print(
                    #    f"id: {video_id} too short, need {minimum_clip_frames} frames"
                    #    f" but annotation shows only {duration[1]}"
                    # )
                    failcnt += 1
                    continue
                obj = VideoMetaData(
                    id=video_id, path=path, label=classname, duration=duration,
                )
                self.data.append(obj)
        print("using {}/{} videos for {}".format(len(self), len(self) + failcnt, mode))

    def _get_base_id(self, path: str) -> str:
        base_id, _ = os.path.splitext(os.path.basename(path))
        return base_id

    def _get_classes_from_root_path(self, video_path: str) -> List[str]:
        classlist = os.listdir(video_path)
        if "test" in classlist:
            classlist.remove("test")
        return classlist

    def _get_paths_from_class(self, video_path: str, classname: str) -> List[str]:
        return sorted(glob.glob(os.path.join(video_path, classname, "*")))

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

    # return number of features
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> VideoData:
        """
        Get tensor of video.
        Returns:
            VideoData containing the clip (n_clip, C, clip_len, H, W)
        """
        obj = self.data[index]
        path = obj.path
        start, end = obj.duration
        frame_indices: List[int] = list(range(start - 1, end - 1))
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
            assert len(frame_indices) == self.clip_len * self.n_clip
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip_tensor = torch.stack([self.spatial_transform(img) for img in clip], 1)
        clip_tensor = self.clipify(clip_tensor, self.clip_len)
        label: int = self.classes.index(obj.label)
        return VideoData(label=label, clip=clip_tensor)
