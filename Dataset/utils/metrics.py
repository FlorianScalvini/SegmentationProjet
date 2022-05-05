import numpy as np
import torch
import torch.nn as torch
import sklearn.metrics as skmetrics


def calculate_area(pred, label, num_classes, ignore_index=255):
    """
    Calculate intersect, prediction and label area
    Args:
        pred (Tensor): The prediction by model.
        label (Tensor): The ground truth of image.
        num_classes (int): The unique number of target classes.
        ignore_index (int): Specifies a target value that is ignored. Default: 255.
    Returns:
        Tensor: The intersection area of prediction and the ground on all class.
        Tensor: The prediction area on all class.
        Tensor: The ground truth area on all class
    """
    if len(pred.shape) == 4:
        pred = torch.squeeze(pred, dim=1)
    if len(label.shape) == 4:
        label = torch.squeeze(label, dim=1)
    if not pred.shape == label.shape:
        raise ValueError('Shape of `pred` and `label should be equal, '
                         'but there are {} and {}.'.format(pred.shape,
                                                           label.shape))
    pred_area = []
    label_area = []
    intersect_area = []
    mask = label != ignore_index

    for i in range(num_classes):
        pred_i = torch.logical_and(pred == i, mask)
        label_i = label == i
        intersect_i = torch.logical_and(pred_i, label_i)
        pred_area.append(torch.sum(torch.Tensor.int(pred_i)))
        label_area.append(torch.sum(torch.Tensor.int(label_i)))
        intersect_area.append(torch.sum(torch.Tensor.int(intersect_i)))

    return intersect_area, pred_area, label_area


def auc_roc(preds, target, num_classes, ignore_index=None):
    """
    Calculate area under the roc curve
    Args:
        preds (Tensor): The prediction by model on testset, of shape (N,C,H,W) .
        target (Tensor): The ground truth of image.   (N,1,H,W)
        num_classes (int): The unique number of target classes.
        ignore_index (int): Specifies a target value that is ignored. Default: 255.
    Returns:
        auc_roc(float): The area under roc curve
    """
    if ignore_index or len(np.unique(target)) > num_classes:
        raise RuntimeError('labels with ignore_index is not supported yet.')

    if len(target.shape) != 4:
        raise ValueError(
            'The shape of target is not 4 dimension as (N, C, H, W), it is {}'.
            format(target.shape))

    if len(preds.shape) != 4:
        raise ValueError(
            'The shape of preds is not 4 dimension as (N, C, H, W), it is {}'.
            format(preds.shape))

    N, C, H, W = preds.shape
    preds = np.transpose(preds, (1, 0, 2, 3))
    preds = preds.reshape([C, N * H * W]).transpose([1, 0])

    target = np.transpose(target, (1, 0, 2, 3))
    target = target.reshape([1, N * H * W]).squeeze()

    if not preds.shape[0] == target.shape[0]:
        raise ValueError('length of `logit` and `label` should be equal, '
                         'but they are {} and {}.'.format(preds.shape[0],
                                                          target.shape[0]))

    if num_classes == 2:
        auc = skmetrics.roc_auc_score(target, preds[:, 1])
    else:
        auc = skmetrics.roc_auc_score(target, preds, multi_class='ovr')

    return auc


def mean_iou(intersect_area, pred_area, label_area):
    """
    Calculate iou.
    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.
    Returns:
        np.ndarray: iou on all classes.
        float: mean iou of all classes.
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    union = pred_area + label_area - intersect_area
    class_iou = []
    for i in range(len(intersect_area)):
        if union[i] == 0:
            iou = 0
        else:
            iou = intersect_area[i] / union[i]
        class_iou.append(iou)
    miou = np.mean(class_iou)
    return np.array(class_iou), miou


def dice(intersect_area, pred_area, label_area):
    """
    Calculate DICE.
    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.
    Returns:
        np.ndarray: DICE on all classes.
        float: mean DICE of all classes.
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    union = pred_area + label_area
    class_dice = []
    for i in range(len(intersect_area)):
        if union[i] == 0:
            vdice = 0
        else:
            vdice = (2 * intersect_area[i]) / union[i]
        class_dice.append(vdice)
    mdice = np.mean(class_dice)
    return np.array(class_dice), mdice


# This is a deprecated function, please use class_measurement function.
def accuracy(intersect_area, pred_area):
    """
    Calculate accuracy
    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes..
        pred_area (Tensor): The prediction area on all classes.
    Returns:
        np.ndarray: accuracy on all classes.
        float: mean accuracy.
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    class_acc = []
    for i in range(len(intersect_area)):
        if pred_area[i] == 0:
            acc = 0
        else:
            acc = intersect_area[i] / pred_area[i]
        class_acc.append(acc)
    macc = np.sum(intersect_area) / np.sum(pred_area)
    return np.array(class_acc), macc


def class_measurement(intersect_area, pred_area, label_area):
    """
    Calculate accuracy, calss precision and class recall.
    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.
    Returns:
        float: The mean accuracy.
        np.ndarray: The precision of all classes.
        np.ndarray: The recall of all classes.
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()

    mean_acc = np.sum(intersect_area) / np.sum(pred_area)
    class_precision = []
    class_recall = []
    for i in range(len(intersect_area)):
        precision = 0 if pred_area[i] == 0 \
            else intersect_area[i] / pred_area[i]
        recall = 0 if label_area[i] == 0 \
            else intersect_area[i] / label_area[i]
        class_precision.append(precision)
        class_recall.append(recall)

    return mean_acc, np.array(class_precision), np.array(class_recall)


def kappa(intersect_area, pred_area, label_area):
    """
    Calculate kappa coefficient
    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes..
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.
    Returns:
        float: kappa coefficient.
    """
    intersect_area = intersect_area.numpy().astype(np.float64)
    pred_area = pred_area.numpy().astype(np.float64)
    label_area = label_area.numpy().astype(np.float64)
    total_area = np.sum(label_area)
    po = np.sum(intersect_area) / total_area
    pe = np.sum(pred_area * label_area) / (total_area * total_area)
    rkappa = (po - pe) / (1 - pe)
    return rkappa
