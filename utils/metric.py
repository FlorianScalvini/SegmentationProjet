import torch
import numpy as np
import torch.nn.functional as F

def kappa(aInter, aPreds, aLabels):
    """
      Calculate kappa coefficient
    :param aInter: The intersection area of prediction and ground truth on all classes..
    :param aPreds: The prediction area on all classes.
    :param aLabels: The ground truth area on all classes.
    :return: kappa coefficient
    """
    total_area = aLabels.sum()
    po = aInter.sum() / total_area
    pe = (aPreds * aLabels).sum() / (total_area * total_area)
    kappa = (po - pe) / (1 - pe)
    return kappa.item()


def diceLoss(aInter, aPreds, aLabels, smooth=1.0):
    loss = 1 - ((2. * aInter.sum() + smooth) /
                (aPreds.sum() + aLabels.sum() + smooth))
    return loss.item()


def class_measurement(aInter, aPreds, aLabels):
    mean_acc = aInter.sum() / aPreds.sum()
    class_precision = aInter / aPreds
    class_recall = aInter / aLabels
    return mean_acc.item(), class_precision, class_recall


def meanIoU(aInter, aPreds, aLabels, ignore_label=None):
    if ignore_label is None and ignore_label in range(aPreds.shape[0]):
        valid_class = aPreds.shape[0] - 1
    else:
        valid_class = aPreds.shape[0]
    union = aPreds + aLabels - aInter
    class_iou = aInter / union
    miou = class_iou.sum() / valid_class
    return class_iou, miou.item()


def calculate_area(prediction, target, classes):
    one_hot_preds = F.one_hot(prediction, classes).transpose(1,3)
    one_hot_targets = F.one_hot(target, classes).transpose(1,3)
    intersection = (one_hot_preds*one_hot_targets).sum(dim=(0,2,3))
    pPreds = one_hot_preds.sum(dim=(0, 2, 3))
    pTarget = one_hot_targets.sum(dim=(0, 2, 3))
    return intersection, pPreds, pTarget