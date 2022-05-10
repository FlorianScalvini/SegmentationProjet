import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import sklearn.metrics as skmetrics

def calculate_area(prediction, target, classes):
    def make_one_hot(labels, classes):
        one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(
            labels.device)
        target = one_hot.scatter_(1, labels.data, 1)

        return target
    one_hot_preds = make_one_hot(prediction, classes)
    one_hot_targets = make_one_hot(target, classes)
    intersection = (one_hot_preds*one_hot_targets).sum(dim=(2,3)).numpy()
    pPreds = one_hot_preds.sum(dim=(0, 2, 3)).numpy()
    pTarget = one_hot_targets.sum(dim=(0, 2, 3)).numpy()
    return intersection, pPreds, pTarget


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        aInter, aPreds, aLabels = calculate_area(preds, targets)
        loss = 1 - ((2. * aInter.sum() + self.smooth) /
                    (aPreds.sum() + aLabels.sum() + self.smooth))
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss



