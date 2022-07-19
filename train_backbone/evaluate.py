from tqdm import tqdm
from utils.metric import *
from utils.loss import loss_computation
np.set_printoptions(suppress=True)


def evaluate(model, eval_loader, num_classes, device, criterion=None, precision='fp32', print_detail=True,
             auc_roc=False, ignore_labels=None):
    model.eval()
    epoch_loss = 0.0
    class_hist = torch.zeros(num_classes)
    target_hist = torch.zeros(num_classes)
    tbar = tqdm(eval_loader, ncols=130, position=0, leave=True)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tbar):

            data = data.to(device)
            target = target.to(device)

            if precision == 'fp16':
                with torch.cuda.amp.autocast(enabled=True):
                    preds = model(data)
                    if criterion is not None:
                        loss = criterion(preds, target)
                        epoch_loss += loss.sum()
            else:
                preds = model(data)
                if criterion is not None:
                    loss = criterion(preds, target)
                    epoch_loss += loss.sum()
            if isinstance(preds, tuple):
                preds = preds[0]
            pred = torch.argmax(preds, dim=1, keepdim=True).squeeze()
            class_hist += torch.histc(pred, num_classes, min=0, max=num_classes-1).to(int)
            target_hist += torch.histc(target, num_classes, min=0, max=num_classes-1).to(int)
    epoch_loss = (epoch_loss / (len(eval_loader) * eval_loader.loader.batch_size)).item()
    class_acc = class_hist / target_hist
    mean_acc = class_acc.mean()
    log = {
        'loss': epoch_loss,
        'mean_acc': mean_acc.cpu().numpy(),
        'class_acc': class_acc.cpu().numpy()
    }
    return log


if __name__ == "__main__":
    print("")
