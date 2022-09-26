import torch
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F


class FocalLoss(_WeightedLoss):

    def __init__(self, weight=None, gamma=2, reduction='mean'):

        super(FocalLoss, self).__init__(weight=weight, reduction=reduction)

        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction=self.reduction, weight=self.weight)
        p_t = torch.exp(-bce_loss)
        loss = (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class LabelSmoothingBCEWithLogitsLoss(_WeightedLoss):

    def __init__(self, weight=None, smoothing_factor=0.0, reduction='mean'):

        super(LabelSmoothingBCEWithLogitsLoss, self).__init__(weight=weight, reduction=reduction)

        self.smoothing_factor = smoothing_factor
        self.weight = weight
        self.reduction = reduction

    def _smooth_labels(self, targets):

        with torch.no_grad():
            targets = targets * (1.0 - self.smoothing_factor) + 0.5 * self.smoothing_factor

        return targets

    def forward(self, inputs, targets):

        targets = self._smooth_labels(targets)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weight)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss