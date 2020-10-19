import json
from typing import Any, Dict, Mapping, Tuple

from . import spatial_transforms, temporal_transforms


class SpatialTransformRepository:
    def __init__(self, mean: float, std: float):
        self._SPATIAL_TRANSFORM_MAPPING: Mapping[str, Tuple[Any, Dict]] = {
            "resize": (spatial_transforms.Resize, {}),
            "resizedcrop": (spatial_transforms.RandomResizedCrop, {}),
            "horizontalflip": (spatial_transforms.RandomHorizontalFlip, {}),
            "grayscale": (spatial_transforms.RandomGrayscale, {}),
            "colorjitter": (spatial_transforms.ColorJitter, {}),
            "totensor": (spatial_transforms.ToTensor, {}),
            "normalize": (spatial_transforms.Normalize, {"mean": mean, "std": std}),
        }

    def get_transform_obj(self, path: str) -> spatial_transforms.Compose:
        with open(path) as f:
            obj = json.load(f)

        transforms = []
        for transform_obj in obj["spatial_transforms"]:
            transform_name = transform_obj["name"]
            if transform_name not in self._SPATIAL_TRANSFORM_MAPPING:
                raise ValueError(f"spatial transform name '{transform_name}' invalid.")
            transform_cls, default_kwargs = self._SPATIAL_TRANSFORM_MAPPING[
                transform_name
            ]

            transform_kwargs = transform_obj["args"]

            transforms.append(transform_cls(**default_kwargs, **{**transform_kwargs}))
        return spatial_transforms.Compose(transforms)


class TemporalTransformRepository:
    def __init__(self):
        self._TEMPORAL_TRANSFORM_MAPPING: Mapping[str, Tuple[Any, Dict]] = {
            "subsampling": (temporal_transforms.TemporalSubsampling, {}),
            "randomcrop": (temporal_transforms.TemporalRandomCrop, {}),
        }

    def get_transform_obj(self, path: str) -> temporal_transforms.Compose:
        with open(path) as f:
            obj = json.load(f)

        transforms = []
        for transform_obj in obj["temporal_transforms"]:
            transform_name = transform_obj["name"]
            if transform_name not in self._TEMPORAL_TRANSFORM_MAPPING:
                raise ValueError(f"temporal transform name '{transform_name}' invalid.")
            transform_cls, default_kwargs = self._TEMPORAL_TRANSFORM_MAPPING[
                transform_name
            ]

            transform_kwargs = transform_obj["args"]
            transforms.append(transform_cls(**default_kwargs, **{**transform_kwargs}))
        return temporal_transforms.Compose(transforms)
