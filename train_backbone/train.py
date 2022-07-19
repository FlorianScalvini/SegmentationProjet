import torch
import torch.nn as nn
import datetime
import time
import os
import tqdm
import logging
import utils.helpers as helpers
from evaluate import evaluate
from utils.DataPrefetcher import DataPrefetcher


def loss_computation(logits_list, labels, criterion, coef):
    len_logits = len(logits_list)
    if len_logits != len(coef):
        raise ValueError("Different number of logits than loss coef. This model requiert " + str(len(logits_list)) + " coeffients")
    loss = 0
    for i in range(len(logits_list)):
        logits = logits_list[i]
        loss += criterion(logits, labels)
    return loss

class TrainerClassification():
    def __init__(self, model, loss, optimizer, scheduler, train_loader, lossCoef, val_loader=None,
                 epochs=100, early_stopping=None, device='cpu', val_per_epochs=10,
                 save_dir="./saved/backbone/", ignore_label=None, *args, **kwargs):
        self.model = model
        self.criterion = loss
        self.lossCoef = lossCoef
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.start_epoch = 1
        self.early_stoping = early_stopping
        self.save_period = 10
        self.not_improved_count = 0
        self.improved = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self.epochs = epochs
        # SETTING THE DEVICE
        self.device = self._get_available_devices(device)
        self.model.to(self.device)
        self.scaler = None
        self.save_dir = save_dir + datetime.now().strftime("%m_%d-%H%M_%S") + "//"
        helpers.create_path(self.save_dir)
        self.num_classes = 33
        self.ignore_labels = ignore_label
        self.metric = "mean_acc"
        if self.device ==  torch.device('cpu'): prefetch = False
        self.precision = 'fp32'
        if self.precision == 'fp16':
            print('use AMP to train.')
            self.scaler = torch.cuda.amp.GradScaler()
        torch.backends.cudnn.benchmark = True
        self.train_loader = DataPrefetcher(train_loader, device=self.device)
        self.val_loader = DataPrefetcher(val_loader, device=self.device)
        #writer = SummaryWriter()


    def _train_epoch(self):
        tbar = tqdm(self.train_loader, ncols=130, position=0, leave=True)
        epoch_loss = 0.0
        class_hist = torch.zeros(self.num_classes)
        target_hist = torch.zeros(self.num_classes)
        for batch_idx, (data, target) in enumerate(tbar):
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    preds = self.model(data)
                    loss = loss_computation(logits_list=preds, labels=target, criterion=self.criterion,
                                            coef=self.lossCoef).sum()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer=self.optimizer)
                self.scaler.update()
            else:
                preds = self.model(data)
                loss = loss_computation(logits_list=preds, labels=target, criterion=self.criterion,
                                        coef=self.lossCoef).sum()
                loss.backward()
            epoch_loss += loss
            pred = torch.argmax(preds, dim=1, keepdim=True).squeeze()
            class_hist += torch.histc(pred, self.num_classes, min=0, max=self.num_classes-1).to(int)
            target_hist += torch.histc(target, self.num_classes, min=0, max=self.num_classes-1).to(int)

        epoch_loss = (epoch_loss / (len(self.train_loader) * self.train_loader.loader.batch_size)).item()
        class_acc = class_hist / target_hist
        mean_acc = class_acc.mean()
        log = {
            'loss': epoch_loss,
            'mean_acc': mean_acc.cpu().numpy(),
            'class_acc': class_acc.cpu().numpy()
        }
        return log


    def train(self):
        mnt_best = 0
        start = time.time()
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_log = self._train_epoch()
            print(f"Epoch :{epoch} \n "
                  f"Train : loss={train_log['loss']} mAcc:{train_log['mean_acc']}")
            val_log = evaluate(model=self.model, eval_loader=self.val_loader, device=self.device,
                               num_classes=self.val_loader.dataset.num_classes, criterion=self.criterion,
                               precision=self.precision, print_detail=False)
            print(f"Val   : loss={val_log['loss']} mAcc:{val_log['mean_acc']}")
            self.scheduler.step(val_log['loss'])
            self.improved = (val_log[self.metric] > mnt_best)
            if self.improved:
                mnt_best = val_log[self.metric]
                self.not_improved_count = 0
            else:
                self.not_improved_count += 1
                if self.early_stoping is not None and self.not_improved_count > self.early_stoping:
                    print(f'\nPerformance didn\'t improve for {self.early_stoping} epochs')
                    print('Training Stoped')
                    break
            # SAVE CHECKPOINT
            if self.improved or self.epochs % self.save_period == 0:
                self._save_checkpoint(epoch, val_log[self.metric],save_best=self.improved)

        time_elapsed = time.time() - start
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(mnt_best))
        return

    def _save_checkpoint(self, epoch, metric_value, save_best=False):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metric': metric_value
        }
        filename = os.path.join(self.save_dir, f'checkpoint-epoch{epoch}.pth')
        print(f'\nSaving a checkpoint: {filename} ...')
        torch.save(state, filename)

        if save_best:
            filename = os.path.join(self.save_dir, f'best_model.pth')
            torch.save(state, filename)
            print("Saving current best: best_model.pth")