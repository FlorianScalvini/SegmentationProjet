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
    num_classes = aPreds.shape()[0]
    aInter = aInter.numpy()
    aPreds = aPreds.numpy()
    aLabels = aLabels.numpy()
    mean_acc = np.sum(aInter) / np.sum(aPreds)
    class_precision = []
    class_recall = []
    for i in range(num_classes):
        precision = 0 if aPreds[i] == 0 else aInter[i] / aPreds[i]
        recall = 0 if aLabels[i] == 0 else aInter[i] / aLabels[i]
        class_precision.append(precision)
        class_recall.append(recall)
    return mean_acc.item(), np.array(class_precision), np.array(class_recall)


def meanIoU(aInter, aPreds, aLabels, ignore_label=None):
    if ignore_label is not None and not isinstance(ignore_label, list):
        ignore_label = list(ignore_label)
    num_classes = aPreds.shape()[0]
    aInter = aInter.numpy()
    aPreds = aPreds.numpy()
    aLabels = aLabels.numpy()
    union = aPreds + aLabels - aInter
    class_iou = aInter / union
    for i in range(num_classes):
        if ignore_label is not None and i in ignore_label:
            continue
        else:
            class_iou.append(aInter[i] / union[i])
    miou = class_iou.mean()
    return class_iou.item(), miou.item()


def calculate_area(prediction, target, classes):
    one_hot_preds = F.one_hot(prediction, classes).transpose(1,3)
    one_hot_targets = F.one_hot(target, classes).transpose(1,3)
    intersection = (one_hot_preds*one_hot_targets).sum(dim=(2,3)).cpu().numpy()
    pPreds = one_hot_preds.sum(dim=(0, 2, 3)).cpu().numpy()
    pTarget = one_hot_targets.sum(dim=(0, 2, 3)).cpu().numpy()
    return intersection, pPreds, pTarget