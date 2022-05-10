import numpy as np
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.metric import *

np.set_printoptions(suppress=True)


def evaluate(model, eval_loader, num_classes, precision='fp32', print_detail=True, auc_roc=False, ignore_labels=None):

    model.eval()
    logits_all = None
    label_all = None

    batch_start = time.time()
    tbar = tqdm(len(eval_loader), ncols=130)
    intersect = torch.zeros(num_classes)
    pred_area = torch.zeros(num_classes)
    label_area = torch.zeros(num_classes)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tbar):
            label = label.astype('int64')
            ori_shape = label.shape[-2:]
            if precision == 'fp16':
                with torch.cuda.amp.autocast(enabled=True):
                    preds = model(data)
            else:
                preds = model(data)
            if isinstance(preds, tuple):
                preds = preds[0]
            pred = torch.argmax(preds, dim=1, keepdim=True)
            inter, pPred, pTarget = calculate_area(pred, label, preds.shape()[1])

            intersect += intersect
            pred_area += pPred
            label_area += pTarget
            tbar.update()

    class_iou, miou = meanIoU(aInter=intersect, aPreds=pred_area, aLabels=label_all, ignore_label=None)
    acc, class_precision, class_recall = class_measurement(aInter=intersect, aPreds=pred_area, aLabels=label_all)
    kap = kappa(aInter=intersect, aPreds=pred_area, aLabels=label_all)
    return miou, acc, class_iou, class_precision, kap
