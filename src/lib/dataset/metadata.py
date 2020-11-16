import enum
import glob
import io
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import h5py
import torch
from lib.dataset import spatial_transforms, temporal_transforms
from PIL import Image
from torch.utils.data import Dataset


class Mode(enum.Enum):
    TRAIN = "training"
    VALIDATION = "validation"
    TEST = "testing"

    @staticmethod
    def from_str(mode_str: str) -> "Mode":
        if mode_str == "training":
            return Mode.TRAIN
        elif mode_str == "validation":
            return Mode.VALIDATION
        elif mode_str == "testing":
            return Mode.TEST
        else:
            raise NotImplementedError


@dataclass
class VideoMetadata:
    id: str
    label: str
    mode: Mode
    path: str
    duration: Tuple[int, int]


class MetadataRepository:
    def __init__(self, root_path: str, mode: Mode):
        self.root_path = root_path
        self.mode = mode

    def get_all(self, path: str) -> List[VideoMetadata]:
        with open(path, "r") as f:
            obj = json.load(f)
        ann: Dict = obj["database"]
        data: List[VideoMetadata] = []
        for video_id, annotation in ann.items():
            split: str = annotation["subset"]
            mode = Mode.from_str(split)
            if mode != self.mode:
                continue
            label: str = annotation["annotations"].get("label", "testing")
            path = self.build_path(label, video_id)
            if not os.path.exists(path):
                print(f"id: {video_id} does not exist as path")
                continue
            if "segment" not in annotation["annotations"]:
                print(f"id: {video_id} has no segment label ({mode})")
                continue
            segment: List[int] = annotation["annotations"]["segment"]
            duration: Tuple[int, int] = (segment[0], segment[1])
            obj = VideoMetadata(
                id=video_id, label=label, mode=mode, path=path, duration=duration,
            )
            data.append(obj)
        return data

    def build_path(self, label: str, id: str) -> str:
        return os.path.join(self.root_path, label, f"{id}.hdf5")
