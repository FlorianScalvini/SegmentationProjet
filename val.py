import numpy as np
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from paddleseg.utils import metrics, TimeAverager, calculate_eta, logger, progbar
from paddleseg.core import infer

np.set_printoptions(suppress=True)


def evaluate(model, eval_loader, precision='fp32', print_detail=True, auc_roc=False):

    model.eval()
    total_iters = len(eval_loader)
    intersect_area_all = torch.zeros()
    pred_area_all = torch.zeros()
    label_area_all = torch.zeros()
    logits_all = None
    label_all = None

    if print_detail:
        logger.info("Start evaluating (total_samples: {}, total_iters: {})...".
                    format(len(eval_dataset), total_iters))
    #TODO(chenguowei): fix log print error with multi-gpus
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    batch_start = time.time()
    tbar = tqdm(eval_loader, ncols=130)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tbar):
            reader_cost_averager.record(time.time() - batch_start)
            label = label.astype('int64')
            ori_shape = label.shape[-2:]
            if precision == 'fp16':
                with torch.cuda.amp.autocast(enabled=True):
                    preds = model(data)
            else:
                preds = model(data)
            if isinstance(preds, tuple):
                preds = preds[0]
            #pred = torch.argmax(preds, dim=1, keepdim=True)

            meanIOU(pred, target, ignore_labels)

            intersect_area_all = intersect_area_all + intersect_area
            pred_area_all = pred_area_all + pred_area
            label_area_all = label_area_all + label_area

            if auc_roc:
                logits = F.softmax(logits, axis=1)
                if logits_all is None:
                    logits_all = logits.numpy()
                    label_all = label.numpy()
                else:
                    logits_all = np.concatenate(
                        [logits_all, logits.numpy()])  # (KN, C, H, W)
                    label_all = np.concatenate([label_all, label.numpy()])

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(label))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            progbar_val.update(iter + 1, [('batch_cost', batch_cost), ('reader cost', reader_cost)])
            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

    metrics_input = (intersect_area_all, pred_area_all, label_area_all)
    class_iou, miou = metrics.mean_iou(*metrics_input)
    acc, class_precision, class_recall = metrics.class_measurement(
        *metrics_input)
    kappa = metrics.kappa(*metrics_input)
    class_dice, mdice = metrics.dice(*metrics_input)

    if auc_roc:
        auc_roc = metrics.auc_roc(logits_all, label_all, num_classes=num_classes)
        auc_infor = ' Auc_roc: {:.4f}'.format(auc_roc)

    if print_detail:
        infor = "[EVAL] #Images: {} mIoU: {:.4f} Acc: {:.4f} Kappa: {:.4f} Dice: {:.4f}".format(len(eval_dataset), miou, acc, kappa, mdice)
        infor = infor + auc_infor if auc_roc else infor
        logger.info(infor)
        logger.info("[EVAL] Class IoU: \n" + str(np.round(class_iou, 4)))
        logger.info("[EVAL] Class Precision: \n" + str(
            np.round(class_precision, 4)))
        logger.info("[EVAL] Class Recall: \n" + str(np.round(class_recall, 4)))
    return miou, acc, class_iou, class_precision, kappa
