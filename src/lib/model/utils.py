import argparse

from torch import nn

from .dpc import DPC, DPCSettings, DPCSettingsRepository
from .dpc_loss import DPCLoss


def get_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "dpc":
        settings: DPCSettings = DPCSettingsRepository().get_settings(args.config_json)
        model = DPC(
            settings.input_size,
            settings.hidden_size,
            settings.kernel_size,
            settings.num_layers,
            settings.n_clip,
            settings.pred_step,
            settings.dropout,
        )
    else:
        raise NotImplementedError(f"model type {args.model} not implemented")
    return model


def get_criterion(args: argparse.Namespace) -> nn.Module:
    if args.model == "dpc":
        criterion = DPCLoss()
    else:
        raise NotImplementedError(f"loss type {args.model} not implemented")
    return criterion
