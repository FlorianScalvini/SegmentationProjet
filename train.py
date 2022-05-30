import time
from tqdm import *
import torch.utils.data
from utils.metric import *
from datetime import *
from transform import *
from utils.DataPrefetcher import DataPrefetcher
import os
import logging
from val import evaluate



def loss_computation(logits_list, labels, loss, coef):
    len_logits = len(logits_list)
    if len_logits != len(coef):
        raise ValueError("Different number of logits than loss coef")
    loss_list = []
    for i in range(len(logits_list)):
        logits = logits_list[i]
        loss_list.append(loss(logits, labels) * coef[i])
    return loss_list


class Trainer():
    def __init__(self, model, loss, optimizer, scheduler, train_loader, lossCoef, val_loader=None,
                 train_logger=None, epochs=100, early_stopping=None, devices='cpu', val_per_epochs=10,
                 save_dir="./saved/", *args, **kwargs):

        self.model = model
        self.loss = loss
        self.lossCoef = lossCoef
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_logger = train_logger
        self.start_epoch = 1
        self.early_stoping = early_stopping
        if val_loader is not None:
            self.val_per_epochs = val_per_epochs
        self.save_period = 10
        self.improved = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self.epochs = epochs
        # SETTING THE DEVICE
        self.device, _ = self._get_available_devices(devices)
        self.model.to(self.device)
        self.scaler = None
        self.save_dir = save_dir + datetime.now().strftime("%m_%d__%H_%M_%S") + "/"
        self.total_loss = 0.
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0
        if self.device ==  torch.device('cpu'): prefetch = False
        self.precision = 'fp32'
        if self.precision == 'fp16':
            print('use AMP to train.')
            self.scaler = torch.cuda.amp.GradScaler()
        torch.backends.cudnn.benchmark = True

        avg_loss = 0.0
        avg_loss_list = []
        best_mean_iou = -1.0
        best_model_iter = -1


    def _get_available_devices(self, device='gpu', n_gpu=0):
        if device == 'gpu':
            sys_gpu = torch.cuda.device_count()
            if sys_gpu == 0:
                self.logger.warning('No GPUs detected, using the CPU')
                n_gpu = 0
            elif n_gpu > sys_gpu:
                self.logger.warning(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
                n_gpu = sys_gpu
            device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
            self.logger.info(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
            available_gpus = list(range(n_gpu))
        else:
            device = torch.device('cpu')
            available_gpus = list()
        return device, available_gpus

    def _train_epoch(self):
        num_classes = self.model.num_classes
        self.model.train()
        self._reset_metrics()
        total_loss = 0.0
        intersect = torch.zeros(num_classes)
        pred_area = torch.zeros(num_classes)
        label_area = torch.zeros(num_classes)
        tbar = tqdm(self.train_loader, ncols=130)
        for batch_idx, (data, target) in enumerate(tbar):
            target = target.astype('int64')
            self.optimizer.zero_grad()
            # data, target = data.to(self.device), target.to(self.device)
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    preds = self.model(data)
                    loss = loss_computation(logits_list=preds, labels=target, loss=self.loss, coef=self.lossCoef)
                    total_loss += loss.sum()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer=self.optimizer)
                self.scaler.update()
            else:
                preds = self.model(data)
                total_loss = loss_computation(logits_list=preds, labels=target, loss=self.loss, coef=self.lossCoef)
                total_loss = loss.sum()
                loss.backward()
                self.optimizer.step()
            if isinstance(preds, tuple):
                preds = preds[0]
            pred = torch.argmax(preds, dim=1, keepdim=True)
            inter, pPred, pTarget = calculate_area(pred, target, preds.shape()[1])
            intersect += intersect
            pred_area += pPred
            label_area += pTarget
            tbar.update()

        # RETURN LOSS & METRICS
        class_iou, miou = meanIoU(aInter=intersect, aPreds=pred_area, aLabels=label_area, ignore_label=None)
        acc, class_precision, class_recall = class_measurement(aInter=intersect, aPreds=pred_area, aLabels=label_area)
        kap = kappa(aInter=intersect, aPreds=pred_area, aLabels=label_area)
        log = {
            'loss': self.total_loss / (len(self.train_loader) * self.train_loader.,
            'miou': miou,
            'class_iou': class_iou,
            'class_precision': class_precision,
            'kappa': kap
        }
        return log

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            results = self._train_epoch(epoch)
            if self.val_loader is not None and epoch % self.val_per_epochs == 0:
                results = evaluate(model=self.model, eval_loader=self.val_loader, num_classes=self.val_loader.numberClasses, precision=self.precision, print_detail=False)
                # LOGGING INFO
                self.logger.info(f'\n ## Info for epoch {epoch} ## ')
                for k, v in results.items():
                    self.logger.info(f'         {str(k):15s}: {v}')

            if self.train_logger is not None:
                log = {'epoch': epoch, **results}
                self.train_logger.add_entry(log)

            if self.mnt_mode != 'off' and epoch % self.config['trainer']['val_per_epochs'] == 0:
                try:
                    if self.mnt_mode == 'min':
                        self.improved = (log[self.mnt_metric] < self.mnt_best)
                    else:
                        self.improved = (log[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    self.logger.warning(f'The metrics being tracked ({self.mnt_metric}) has not been calculated. Training stops.')
                    break

                if self.improved:
                    self.mnt_best = log[self.mnt_metric]
                    self.not_improved_count = 0
                else:
                    self.not_improved_count += 1

                if self.not_improved_count > self.early_stoping:
                    self.logger.info(f'\nPerformance didn\'t improve for {self.early_stoping} epochs')
                    self.logger.warning('Training Stoped')
                    break

            # SAVE CHECKPOINT
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=self.improved)

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def _reset_metrics(self):
        self.total_loss = 0.
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0


    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.save_dir, f'checkpoint-epoch{epoch}.pth')
        self.logger.info(f'\nSaving a checkpoint: {filename} ...')
        torch.save(state, filename)

        if save_best:
            filename = os.path.join(self.save_dir, f'best_model.pth')
            torch.save(state, filename)
            self.logger.info("Saving current best: best_model.pth")


    def _resume_checkpoint(self, resume_path):
        self.logger.info(f'Loading checkpoint : {resume_path}')
        checkpoint = torch.load(resume_path)

        # Load last run info, the model params, the optimizer and the loggers
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.not_improved_count = 0
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(f'Checkpoint <{resume_path}> (epoch {self.start_epoch}) was loaded')

"""
def train(model,
          config,
          precision='fp32'):

    model.train()
    print(model)
    device = 'gpu'


    train_dataset = cityscrape.Cityscapes(root="/media/ubuntu/DATA/Database/leftImg8bit_trainvaltest/",
                                          mode='train', transforms=[Resize((224,244)), Normalize(32,32)])
    val_dataset = cityscrape.Cityscapes(root="/media/ubuntu/DATA/Database/leftImg8bit_trainvaltest/",
                                          mode='val', transforms=[Resize((224,244)), Normalize(32,32)])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2, num_workers=4, drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=2, num_workers=4, drop_last=True)
    Trainer(model=model, loss=loss, config=config, train_logger=train_loader, val_loader=val_loader, precision=precision)
"""