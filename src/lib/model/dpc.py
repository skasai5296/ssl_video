import json
from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models.video import r3d_18


class ClipEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = r3d_18(pretrained=False)
        modules = list(self.model.children())[:-2]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        """
        x : (B, C, T, H, W)
        out : (B, 512, H', W')
        """
        out = self.model(x)
        return out.mean(2)


class ConvGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, kernel_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        self.reset_gate = nn.Conv2d(
            input_size + hidden_size, hidden_size, kernel_size, padding=padding
        )
        self.update_gate = nn.Conv2d(
            input_size + hidden_size, hidden_size, kernel_size, padding=padding
        )
        self.out_gate = nn.Conv2d(
            input_size + hidden_size, hidden_size, kernel_size, padding=padding
        )

    def forward(self, x, h):
        """
        x : (B, input_size, H', W')
        h : (B, hidden_size, H', W')
        """
        if h is None:
            B, C, *spatial_dim = x.size()
            h = torch.zeros([B, self.hidden_size, *spatial_dim], device=x.device)
        input = torch.cat([x, h], dim=1)
        update = torch.sigmoid(self.update_gate(input))
        reset = torch.sigmoid(self.reset_gate(input))
        out = torch.tanh(self.out_gate(torch.cat([x, h * reset], dim=1)))
        new_state = h * (1 - update) + out * update
        return new_state


class ConvGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        cell_list = []
        for i in range(num_layers):
            input_dim = self.input_size if i == 0 else self.hidden_size
            cell = ConvGRUCell(input_dim, self.hidden_size, self.kernel_size)
            name = "ConvGRUCell_{:02d}".format(i)
            setattr(self, name, cell)
            cell_list.append(getattr(self, name))

        self.cell_list = nn.ModuleList(cell_list)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, h=None):
        """
        x : (B, T, input_size, H, W)
        layer_output : (B, T, hidden_size, H, W)
        last_state_list : (B, num_layers, hidden_size, H, W)
        """
        B, T, *_ = x.size()
        if h is None:
            h = [None] * self.num_layers
        current_layer_input = x
        del x

        last_state_list = []

        for idx in range(self.num_layers):
            cell_hidden = h[idx]
            output_inner = []
            for t in range(T):
                cell_hidden = self.cell_list[idx](
                    current_layer_input[:, t, ...], cell_hidden
                )
                cell_hidden = self.dropout(cell_hidden)
                output_inner.append(cell_hidden)

            layer_output = torch.stack(output_inner, dim=1)
            current_layer_input = layer_output
            last_state_list.append(cell_hidden)
        last_state_list = torch.stack(last_state_list, dim=1)
        return layer_output, last_state_list


class DPC(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        num_layers: int,
        n_clip: int,
        pred_step: int,
        dropout: float,
    ):
        super().__init__()
        self.n_clip = n_clip
        self.pred_step = pred_step
        self.cnn = ClipEncoder()
        self.rnn = ConvGRU(input_size, hidden_size, kernel_size, num_layers, dropout)
        self.network_pred = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1, padding=0),
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, flag="full"):
        if flag == "full":
            return self._full_pass(x)
        elif flag == "extract":
            return self._extract_feature(x)

    def _full_pass(self, x):
        """
        x: (B, num_clips, C, clip_len, H, W)
        pred, out : (B, N, hidden_size, H, W)
        """
        B, N, *sizes = x.size()
        x = x.view(B * N, *sizes)
        # out : (B * N, hidden_size, H', W')
        out = self.cnn(x)
        _, D, H, W = out.size()
        # out : (B, N, hidden_size, H', W')
        out = out.view(B, N, D, H, W)

        # hidden: (B, hidden_size, H', W')
        _, hidden = self.rnn(out[:, : self.n_clip - self.pred_step, ...])
        hidden = hidden[:, -1, ...]
        pred = []
        for step in range(self.pred_step):
            # predicted: (B, hidden_size, H', W')
            predicted = self.network_pred(hidden)
            pred.append(predicted)
            _, hidden = self.rnn(self.relu(predicted).unsqueeze(1), hidden.unsqueeze(0))
            hidden = hidden[:, -1, ...]
        # pred: (B, pred_step, hidden_size, H', W')
        pred = torch.stack(pred, 1)
        return pred, out[:, self.n_clip - self.pred_step :, ...]

    # x: (B, num_clips, C, clip_len, H, W)
    # hidden : (B, hidden_size, H, W)
    def _extract_feature(self, x):
        B, N, *sizes = x.size()
        x = x.view(B * N, *sizes)
        # out : (B * N, hidden_size, H', W')
        out = self.cnn(x)
        _, D, H, W = out.size()
        # out : (B, N, hidden_size, H', W')
        out = out.view(B, N, D, H, W)
        # hidden: (B, hidden_size, H', W')
        _, hidden = self.rnn(out[:, : self.n_clip - self.pred_step, ...])
        hidden = hidden[:, -1, ...]
        return hidden


@dataclass
class DPCSettings:
    n_clip: int
    input_size: int
    hidden_size: int
    kernel_size: int
    num_layers: int
    pred_step: int
    dropout: float


class DPCSettingsRepository:
    def get_settings(self, path: str) -> DPCSettings:
        with open(path, "r") as f:
            obj = json.load(f)
        return DPCSettings(
            n_clip=obj["n_clip"],
            input_size=obj["input_size"],
            hidden_size=obj["hidden_size"],
            kernel_size=obj["kernel_size"],
            num_layers=obj["num_layers"],
            pred_step=obj["pred_step"],
            dropout=obj["dropout"],
        )
