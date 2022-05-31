from tqdm import tqdm
from utils.metric import *
from utils.loss import loss_computation
np.set_printoptions(suppress=True)


def evaluate(model, eval_loader, num_classes, loss=None, lossCoef=None,
             precision='fp32', print_detail=True, auc_roc=False, ignore_labels=None):

    model.eval()
    tbar = tqdm(len(eval_loader), ncols=130)
    intersect = torch.zeros(num_classes)
    pred_area = torch.zeros(num_classes)
    label_area = torch.zeros(num_classes)
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tbar):
            if precision == 'fp16':
                with torch.cuda.amp.autocast(enabled=True):
                    preds = model(data)
                    if loss is not None and lossCoef is not None:
                        loss = loss_computation(logits_list=preds, labels=target, loss=loss, coef=lossCoef)
                        total_loss += loss.sum()
            else:
                preds = model(data)
                if loss is not None and lossCoef is not None:
                    loss = loss_computation(logits_list=preds, labels=target, loss=loss, coef=lossCoef)
                    total_loss += loss.sum()
            if isinstance(preds, tuple):
                preds = preds[0]
            pred = torch.argmax(preds, dim=1, keepdim=True)
            inter, pPred, pTarget = calculate_area(pred, target, preds.shape()[1])

            intersect += intersect
            pred_area += pPred
            label_area += pTarget
            tbar.update()

    class_iou, miou = meanIoU(aInter=intersect, aPreds=pred_area, aLabels=label_area, ignore_label=ignore_labels)
    acc, class_precision, class_recall = class_measurement(aInter=intersect, aPreds=pred_area, aLabels=label_area)
    kap = kappa(aInter=intersect, aPreds=pred_area, aLabels=label_area)

    log = {
        'miou': miou,
        'class_iou': class_iou,
        'class_precision': class_precision,
        'kappa': kap
    }
    if loss is not None and lossCoef is not None:
        log['loss'] = total_loss / (len(eval_loader) * eval_loader.batch_size),
    return log
