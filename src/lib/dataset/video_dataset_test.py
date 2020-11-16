import enum
import glob
import io
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import pytest
import torch
from lib.dataset import spatial_transforms, temporal_transforms, video_dataset
from lib.dataset.metadata import VideoMetadata
from lib.dataset.video_dataset import (
    ClipBatch,
    ClipData,
    VideodataRepository,
    collate_data,
)


@pytest.fixture
def sample_clips() -> List[ClipData]:
    return [
        ClipData(label=0, clip=torch.tensor([0, 1, 2, 3])),
        ClipData(label=1, clip=torch.tensor([4, 5, 6, 7])),
        ClipData(label=2, clip=torch.tensor([8, 9, 10, 11])),
        ClipData(label=3, clip=torch.tensor([12, 13, 14, 15])),
    ]


def test_clipify(sample_clips: List[ClipData]):
    clip_batch = collate_data(sample_clips)
    assert torch.equal(clip_batch.label, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(
        clip_batch.clip,
        torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]),
    )


class TestVideoDataRepository:
    pass
    # @pytest.mark.parametrize("tensor, clip_len, n_clip, output")
    # def test_clipify(self, tensor: torch.Tensor, clip_len: int, n_clip: int, output: torch.Tensor):
    #    videodata_repo = VideoDataRepository("", "", "", clip_len, n_clip, 0)
