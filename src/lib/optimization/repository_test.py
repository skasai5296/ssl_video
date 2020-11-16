from typing import Any, Dict, Mapping, Tuple, Type, Union

import pytest
import torch
from pytest_mock import MockerFixture

from .repository import OptimizerRepository, SchedulerRepository


class TestOptimizerRepository:
    def test_get_optimizer(self, mocker: MockerFixture):
        model = mocker.Mock()
        model.parameters.return_value = "mockedparams"
        mocker.patch("torch.optim.SGD", autospec=True)

        optimizer_repo = OptimizerRepository()
        optimizer_repo.get_optimizer(model, "src/default_optimizer.json")

        model.parameters.assert_called_once()
        torch.optim.SGD.assert_called_once_with(  # type: ignore
            "mockedparams", lr=0.001, weight_decay=0.00001
        )


class TestSchedulerRepository:
    def test_get_scheduler(self, mocker: MockerFixture):
        model = mocker.Mock()
        model.parameters.return_value = "mockedparams"
        mocker.patch("torch.optim.lr_scheduler.StepLR", autospec=True)
        optim_mock = mocker.Mock()

        scheduler_repo = SchedulerRepository()
        scheduler_repo.get_scheduler(optim_mock, "src/default_optimizer.json")

        torch.optim.lr_scheduler.StepLR.assert_called_once_with(  # type: ignore
            optim_mock, step_size=int(1e10),
        )
