from tqdm import tqdm
from utils.metric import *
from utils.loss import loss_computation
np.set_printoptions(suppress=True)


def evaluate(model, eval_loader, num_classes, device, criterion=None, precision='fp32', print_detail=True,
             auc_roc=False, ignore_labels=None):
    model.eval()
    intersect = torch.zeros(num_classes).to(device)
    pred_area = torch.zeros(num_classes).to(device)
    label_area = torch.zeros(num_classes).to(device)
    tbar = tqdm(eval_loader, ncols=130, position=0, leave=True)

    total_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tbar):

            data = data.to(device)
            target = target.to(device)

            if precision == 'fp16':
                with torch.cuda.amp.autocast(enabled=True):
                    preds = model(data)
                    if criterion is not None:
                        loss = criterion(preds, target)
                        total_loss += loss.sum()
            else:
                preds = model(data)
                if criterion is not None:
                    loss = criterion(preds, target)
                    total_loss += loss.sum()
            if isinstance(preds, tuple):
                preds = preds[0]
            pred = torch.argmax(preds, dim=1, keepdim=True).squeeze()
            inter, pPred, pTarget = calculate_hist(pred, target, preds.shape[1], ignore_labels=ignore_labels)
            intersect = torch.add(inter, intersect)
            pred_area = torch.add(pPred, pred_area)
            label_area = torch.add(pTarget, label_area)


    class_iou, miou = meanIoU(aInter=intersect, aPreds=pred_area, aLabels=label_area)
    acc, class_precision, class_recall = class_measurement(aInter=intersect, aPreds=pred_area, aLabels=label_area)
    kap = kappa(aInter=intersect, aPreds=pred_area, aLabels=label_area)

    log = {
        'miou': miou,
        'class_iou': class_iou.cpu().numpy(),
        'class_precision': class_precision.cpu().numpy(),
        'kappa': kap
    }
    if criterion is not None:
        log['loss'] = (total_loss / (len(eval_loader) * eval_loader.loader.batch_size)).cpu().numpy()
    return log
