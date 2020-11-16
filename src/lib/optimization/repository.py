import json
from typing import Any, Dict, Mapping, Tuple, Type, Union

from torch import nn, optim
from torch.optim import lr_scheduler

# change to optim.Optimizer for torch >= 1.5.0
OPTIMIZER_TYPE = Any
# TODO: change to Union[lr_scheduler._LRScheduler, lr_scheduler.ReduceLROnPlateau]
SCHEDULER_TYPE = Any


class OptimizerRepository:
    def __init__(self):
        self._OPTIMIZER_MAPPING: Mapping[str, Tuple[Type[OPTIMIZER_TYPE], Dict]] = {
            "sgd": (optim.SGD, {}),
            "adam": (optim.Adam, {}),
        }

    def get_optimizer(self, model: nn.Module, path: str) -> OPTIMIZER_TYPE:
        with open(path) as f:
            obj = json.load(f)

        name = obj["optimizer"]["name"]
        if name not in self._OPTIMIZER_MAPPING:
            raise ValueError(f"optimizer name '{name}' invalid.")
        kwargs: Dict = obj["optimizer"]["args"]
        optimizer_cls, default_kwargs = self._OPTIMIZER_MAPPING[name]
        optimizer = optimizer_cls(model.parameters(), **default_kwargs, **{**kwargs})
        return optimizer


class SchedulerRepository:
    def __init__(self):
        self._SCHEDULER_MAPPING: Mapping[str, Any] = {
            "none": [lr_scheduler.StepLR, {"step_size": int(1e10)}],
            "step": [lr_scheduler.StepLR, {}],
            "plateau": [lr_scheduler.ReduceLROnPlateau, {}],
        }

    def get_scheduler(self, optimizer: OPTIMIZER_TYPE, path: str) -> SCHEDULER_TYPE:
        with open(path) as f:
            obj = json.load(f)

        name = obj["scheduler"]["name"]
        if name not in self._SCHEDULER_MAPPING:
            raise ValueError(f"scheduler name '{name}' invalid.")
        kwargs: Dict = obj["scheduler"]["args"]
        scheduler_cls, default_kwargs = self._SCHEDULER_MAPPING[name]
        scheduler = scheduler_cls(optimizer, **default_kwargs, **{**kwargs})
        return scheduler
