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


def meanIoU(aInter, aPreds, aLabels):
    union = aPreds + aLabels - aInter
    class_iou = aInter / union
    miou = torch.mean(class_iou)
    return class_iou, miou.item()


def calculate_hist(prediction, target, num_classes, ignore_labels=None):
    if ignore_labels is None:
        intersection = prediction[prediction == target]
    else:
        intersection = prediction[torch.logical_and(prediction == target, ignore_labels!=target)]
    preds_intersection = torch.histc(intersection, bins=num_classes, min=0, max=num_classes-1)
    preds_hist= torch.histc(prediction, bins=num_classes, min=0, max=num_classes-1)
    target_hist = torch.histc(target, bins=num_classes, min=0, max=num_classes-1)
    return preds_intersection, preds_hist, target_hist