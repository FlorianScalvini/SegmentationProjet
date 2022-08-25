import torchvision.utils
from tqdm import tqdm
from utils.metric import *
from utils.loss import loss_computation
np.set_printoptions(suppress=True)
import torch.nn as nn
from utils.helpers import *
from utils.utils import *
from torchvision.utils import *
import torch.nn.functional as F

def evaluate(model, eval_loader, num_classes, device, criterion=None, precision='fp32', print_detail=True,
             auc_roc=False, ignore_labels=None, nb_vizimg=4, palette=None):
    model.eval()
    intersect = torch.zeros(num_classes).to(device)
    pred_area = torch.zeros(num_classes).to(device)
    label_area = torch.zeros(num_classes).to(device)
    tbar = tqdm(eval_loader, ncols=130, position=0, leave=True)

    upscale = nn.Upsample(scale_factor=2, mode='nearest')
    total_loss = 0
    val_visual = []


    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(tbar):

            if precision == 'fp16':
                with torch.cuda.amp.autocast(enabled=True):
                    if not model.depth:
                        preds = model(input[0])
                    else:
                        preds = model(*input)
                    if criterion is not None:
                        size = target.shape[-2:]
                        preds = F.interpolate(preds, size=size)
                        loss = criterion(preds, target)
                        total_loss += loss.sum()
            else:
                if not model.depth:
                    preds = model(input[0])
                else:
                    preds = model(*input)
                if criterion is not None:
                    size = target.shape[-2:]
                    preds = F.interpolate(preds, size=size)
                    loss = criterion(preds, target)
                    total_loss += loss.sum()
            if isinstance(preds, tuple):
                preds = preds[0]
            pred = torch.argmax(preds, dim=1, keepdim=True).squeeze()
            inter, pPred, pTarget = calculate_hist(pred, target, preds.shape[1], ignore_labels=ignore_labels)
            intersect = torch.add(inter, intersect)
            pred_area = torch.add(pPred, pred_area)
            label_area = torch.add(pTarget, label_area)

            if len(val_visual) < nb_vizimg:
                size = target.shape[-2:]
                imgs = F.interpolate(input[0], size=size)
                for i in range(imgs.shape[0]):
                    if len(val_visual) >= nb_vizimg:
                        break
                    else:
                        val_visual.append([imgs[i], pred[i], target[i]])


    class_iou, miou = meanIoU(aInter=intersect, aPreds=pred_area, aLabels=label_area)
    acc, class_precision, class_recall = class_measurement(aInter=intersect, aPreds=pred_area, aLabels=label_area)

    log = {
        'miou': miou,
        'class_iou': class_iou.cpu().numpy(),
        'class_precision': class_precision.cpu().numpy(),
    }
    if criterion is not None:
        log['loss'] = (total_loss / (len(eval_loader) * eval_loader.loader.batch_size)).cpu().numpy()


    if len(val_visual) > 0:
        # WRTING & VISUALIZING THE MASKS
        val_img = []
        for d, o, t in val_visual:
            o = labeltoColor(label=o, color_map=palette, num_classes=num_classes, device=device).to('cpu')
            t = labeltoColor(label=t, color_map=palette, num_classes=num_classes, device=device).to('cpu')
            val_img.extend([d.to('cpu'), t, o])
        val_img = torch.stack(val_img, 0)
        val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
        log['image'] = val_img
    return log
