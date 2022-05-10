import torch
import numpy as np


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