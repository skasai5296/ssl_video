import io
from typing import List, Tuple

import h5py
from PIL import Image


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


def get_stats() -> Tuple[List[float], List[float]]:
    """get mean and std for Kinetics"""
    maximum = 255.0
    mean = [110.63666788 / maximum, 103.16065604 / maximum, 96.29023126 / maximum]
    std = [38.7568578 / maximum, 37.88248729 / maximum, 40.02898126 / maximum]
    return mean, std


# def get_transforms(
#    mode,
#    resize: int,
#    clip_len: int,
#    n_clip: int,
#    downsample: int,
#    consistent: bool = True,
# ) -> Tuple[spatial_transforms.Compose, temporal_transforms.Compose]:
#    assert mode in ("train", "val", "extract")
#    mean, std = get_stats()
#    if mode == "train":
#        if consistent:
#            sp_t = spatial_transforms.Compose(
#                [
#                    spatial_transforms.Resize(int(resize * 8 / 7)),
#                    spatial_transforms.RandomResizedCrop(
#                        size=(resize, resize), scale=(0.5, 1.0)
#                    ),
#                    spatial_transforms.RandomHorizontalFlip(),
#                    spatial_transforms.RandomGrayscale(p=0.5),
#                    spatial_transforms.ColorJitter(
#                        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25,
#                    ),
#                    spatial_transforms.ToTensor(),
#                    spatial_transforms.Normalize(mean=mean, std=std),
#                ]
#            )
#        else:
#            sp_t = spatial_transforms.Compose(
#                [
#                    spatial_transforms.Resize(int(resize * 8 / 7)),
#                    spatial_transforms.RandomResizedCrop(
#                        size=(resize, resize), scale=(0.5, 1.0)
#                    ),
#                    spatial_transforms.RandomHorizontalFlip(),
#                    spatial_transforms.RandomGrayscale_Nonconsistent(p=0.5),
#                    spatial_transforms.ColorJitter_Nonconsistent(
#                        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25,
#                    ),
#                    spatial_transforms.ToTensor(),
#                    spatial_transforms.Normalize(mean=mean, std=std),
#                ]
#            )
#        tp_t = temporal_transforms.Compose(
#            [
#                temporal_transforms.TemporalSubsampling(downsample),
#                # temporal_transforms.SlidingWindow(size=clip_len * n_clip, stride=6),
#                temporal_transforms.TemporalRandomCrop(size=clip_len * n_clip),
#            ]
#        )
#    elif mode == "val":
#        sp_t = spatial_transforms.Compose(
#            [
#                spatial_transforms.Resize(resize),
#                spatial_transforms.CenterCrop(size=(resize, resize)),
#                spatial_transforms.ToTensor(),
#                spatial_transforms.Normalize(mean=mean, std=std),
#            ]
#        )
#        tp_t = temporal_transforms.Compose(
#            [
#                temporal_transforms.TemporalSubsampling(downsample),
#                # temporal_transforms.SlidingWindow(size=clip_len * n_clip, stride=6),
#                temporal_transforms.TemporalRandomCrop(size=clip_len * n_clip),
#            ]
#        )
#    elif mode == "extract":
#        sp_t = spatial_transforms.Compose(
#            [
#                spatial_transforms.ToPILImage(),
#                spatial_transforms.Resize(resize),
#                spatial_transforms.CenterCrop(size=(resize, resize)),
#                spatial_transforms.ToTensor(),
#                spatial_transforms.Normalize(mean=mean, std=std),
#            ]
#        )
#        tp_t = temporal_transforms.Compose(
#            [
#                temporal_transforms.TemporalSubsampling(downsample),
#                # temporal_transforms.SlidingWindow(size=clip_len * n_clip, stride=8),
#            ]
#        )
#    return sp_t, tp_t
#
#
# def get_transforms_finetune(
#    mode: str, resize: int, clip_len: int, n_clip: int, downsample: int
# ) -> Tuple[spatial_transforms.Compose, temporal_transforms.Compose]:
#    assert mode in ("train", "val")
#    mean, std = get_stats()
#    if mode == "train":
#        sp_t = spatial_transforms.Compose(
#            [
#                spatial_transforms.Resize(int(resize * 8 / 7)),
#                spatial_transforms.RandomResizedCrop(
#                    size=(resize, resize), scale=(0.5, 1.0)
#                ),
#                spatial_transforms.RandomHorizontalFlip(),
#                spatial_transforms.ToTensor(),
#                spatial_transforms.Normalize(mean=mean, std=std),
#            ]
#        )
#    elif mode == "val":
#        sp_t = spatial_transforms.Compose(
#            [
#                spatial_transforms.Resize(resize),
#                spatial_transforms.CenterCrop(size=(resize, resize)),
#                spatial_transforms.ToTensor(),
#                spatial_transforms.Normalize(mean=mean, std=std),
#            ]
#        )
#    tp_t = temporal_transforms.Compose(
#        [
#            temporal_transforms.TemporalSubsampling(downsample),
#            # temporal_transforms.SlidingWindow(size=clip_len * n_clip, stride=6),
#            temporal_transforms.TemporalRandomCrop(size=clip_len * n_clip),
#        ]
#    )
#    return sp_t, tp_t
#
#
# def collate_fn(datalist: List[Dict]) -> Dict:
#    clips: List[torch.Tensor] = []
#    labels: List[str] = []
#    for data in datalist:
#        clip = data["clip"]
#        label = data["label"]
#        clips.append(clip)
#        labels.append(label)
#    clip_tensor: torch.Tensor = torch.stack(clips)
#    label_tensor: torch.Tensor = torch.tensor(labels, dtype=torch.long)
#    return {"clip": clip_tensor, "label": label_tensor}
