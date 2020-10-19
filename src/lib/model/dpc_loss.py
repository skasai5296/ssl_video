from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Metric:
    loss: float
    accuracy: float

    def __repr__(self) -> str:
        acc_percent = self.accuracy * 100
        return f"loss: {self.loss:.03f}, acc: {acc_percent:.03f}%"


class DPCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred, gt):
        """
        Compute loss (dot product over feature dimension)
        Args:
            pred, gt: torch.Tensor (B, N, C, H, W)
        Returns:
            loss: torch.Tensor, sum of all losses
            losses: loss dict
        """
        assert pred.size() == gt.size()
        B, N, C, H, W = pred.size()
        pred = pred.permute(0, 1, 3, 4, 2).reshape(B * N * H * W, -1)
        gt = gt.permute(2, 0, 1, 3, 4).reshape(-1, B * N * H * W)
        # lossmat: (BNHW, BNHW)
        lossmat = torch.matmul(pred, gt)
        # target: (BNHW)
        target = torch.arange(B * N * H * W, dtype=torch.long, device=pred.device)
        loss = self.criterion(lossmat, target)

        with torch.no_grad():
            # top1: (BNHW)
            top1 = lossmat.argmax(1)
            acc = torch.eq(top1, target).sum().item() / top1.size(0)
        self.metric = Metric(loss=loss.item(), accuracy=acc)
        return loss


class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred, gt):
        """
        Compute loss (dot product over feature dimension)
        Args:
            pred: torch.Tensor (B, num_classes)
            gt: torch.Tensor (B), torch.long
        Returns:
            loss: torch.Tensor, sum of all losses
            losses: loss dict
        """
        loss = self.criterion(pred, gt)

        with torch.no_grad():
            # top1: (BNHW)
            top1 = pred.argmax(1)
            acc = torch.eq(top1, gt).sum().item() / top1.size(0) * 100
        return loss, {"XELoss": loss.item(), "Accuracy (%)": acc}
