import time
from tqdm.auto import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data
from utils.metric import *
from datetime import *
from transform import *
from utils.DataPrefetcher import DataPrefetcher
from utils.loss import loss_computation
import os
import logging
from val import evaluate


class Trainer():
    def __init__(self, model, loss, optimizer, scheduler, train_loader, lossCoef, val_loader=None,
                 epochs=100, early_stopping=None, device='cpu', val_per_epochs=10,
                 save_dir="./saved/", ignore_label=None, *args, **kwargs):

        self.model = model
        self.loss = loss
        self.lossCoef = lossCoef
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.start_epoch = 1
        self.early_stoping = early_stopping
        self.metric = 'miou'
        if val_loader is not None:
            self.val_per_epochs = val_per_epochs
        self.save_period = 10
        self.improved = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self.epochs = epochs
        # SETTING THE DEVICE
        self.device = self._get_available_devices(device)
        self.model.to(self.device)
        self.scaler = None
        self.save_dir = save_dir + datetime.now().strftime("%m_%d__%H_%M_%S") + "/"
        self.total_loss = 0.
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0
        self.ignore_labels = 255
        if self.device ==  torch.device('cpu'): prefetch = False
        self.precision = 'fp32'
        if self.precision == 'fp16':
            print('use AMP to train.')
            self.scaler = torch.cuda.amp.GradScaler()
        torch.backends.cudnn.benchmark = True
        self.train_loader = DataPrefetcher(train_loader, device=self.device)
        self.val_loader = DataPrefetcher(val_loader, device=self.device)
        #writer = SummaryWriter()

    def _get_available_devices(self, device='gpu', n_gpu=0):
        if device == 'gpu':
            sys_gpu = torch.cuda.device_count()
            if sys_gpu == 0:
                self.logger.warning('No GPUs detected, using the CPU')
                n_gpu = 0
            elif n_gpu > sys_gpu:
                self.logger.warning(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
                n_gpu = sys_gpu
            device = torch.device('cuda:0')
            self.logger.info(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
        else:
            device = torch.device('cpu')
        return device

    def _train_epoch(self, epoch=None):
        num_classes = self.model.num_classes
        self.model.train()
        self._reset_metrics()
        total_loss = 0.0
        intersect = torch.zeros(num_classes).to(self.device)
        pred_area = torch.zeros(num_classes).to(self.device)
        label_area = torch.zeros(num_classes).to(self.device)
        tbar = tqdm(self.train_loader, ncols=130, position=0, leave=True)
        for batch_idx, (data, target) in enumerate(tbar):
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(data)
            loss = loss_computation(logits_list=preds, labels=target, criterion=self.loss, coef=self.lossCoef).sum()
            loss.backward()
            total_loss += loss
            if isinstance(preds, tuple) or isinstance(preds, list):
                preds = preds[0]
            pred = torch.argmax(preds, dim=1, keepdim=True).squeeze()
            inter, pPred, pTarget = calculate_hist(pred, target, preds.shape[1], ignore_labels=self.ignore_labels)
            intersect = torch.add(inter, intersect)
            pred_area = torch.add(pPred, pred_area)
            label_area = torch.add(pTarget, label_area)

        # RETURN LOSS & METRICS
        class_iou, miou = meanIoU(aInter=intersect, aPreds=pred_area, aLabels=label_area)
        acc, class_precision, class_recall = class_measurement(aInter=intersect, aPreds=pred_area, aLabels=label_area)
        kap = kappa(aInter=intersect, aPreds=pred_area, aLabels=label_area)
        log = {
            'loss': self.total_loss / (len(self.train_loader) * self.train_loader.loader.batch_size),
            'miou': miou,
            'class_iou': class_iou,
            'class_precision': class_precision,
            'kappa': kap
        }
        return log


    def train(self):
        best_metric = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_log = self._train_epoch()
            val_log = evaluate(model=self.model, eval_loader=self.val_loader, num_classes=self.val_loader.dataset.num_classes, precision=self.precision, print_detail=False)
            self.scheduler.step(val_log['loss'])
            # LOGGING INFO
            self.logger.info(f'\n ## Info for epoch {epoch} ## ')
            for k, v in train_log.items():
                self.logger.info(f'         {str(k):15s}: {v}')

            log = {'epoch': epoch,
                   'train': train_log,
                   'val':  val_log}

            if epoch % self.config['trainer']['val_per_epochs'] == 0:
                self.improved = (val_log[self.metric] > best_metric)

                if self.improved:
                    self.mnt_best = val_log[self.metric]
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
