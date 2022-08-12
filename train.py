from tqdm.auto import tqdm, trange
import torch.utils.data
from utils.metric import *
from datetime import *
from transform import *
from utils.DataPrefetcher import DataPrefetcher
from utils.loss import loss_computation
import os
import logging
from evaluate import evaluate
import utils.helpers as helpers

class Trainer():
    def __init__(self, model, loss, optimizer, scheduler, train_loader, lossCoef, val_loader=None,
                 epochs=100, early_stopping=None, device='cpu', val_per_epochs=10,
                 save_dir="./saved/", ignore_label=None, *args, **kwargs):

        self.model = model
        self.criterion = loss
        self.lossCoef = lossCoef
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.start_epoch = 1
        self.early_stoping = early_stopping
        self.metric = 'miou'
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

    def _train_epoch(self, epoch=None, rgbd_train=False):
        num_classes = self.model.num_classes
        self.model.train()
        total_loss = 0.0
        intersect = torch.zeros(num_classes).to(self.device)
        pred_area = torch.zeros(num_classes).to(self.device)
        label_area = torch.zeros(num_classes).to(self.device)
        tbar = tqdm(self.train_loader, ncols=130, position=0, leave=True)
        for batch_idx, (img, depth_img, target) in enumerate(tbar):
            img = img.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    if rgbd_train:
                        preds = self.model(img, depth_img)
                    else:
                        preds = self.model(img)
                    loss = loss_computation(logits_list=preds, labels=target, criterion=self.criterion,
                                            coef=self.lossCoef).sum()
                    loss = loss.sum()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer=self.optimizer)
                self.scaler.update()
            else:
                if rgbd_train:
                    preds = self.model(img, depth_img)
                else:
                    preds = self.model(img)
                loss = loss_computation(logits_list=preds, labels=target, criterion=self.criterion,
                                        coef=self.lossCoef).sum()
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
        log = {
            'loss': (total_loss / (len(self.train_loader) * self.train_loader.loader.batch_size)).item(),
            'miou': miou,
            'class_iou': class_iou.cpu().numpy(),
            'class_precision': class_precision.cpu().numpy(),
        }
        return log


    def train(self):
        self.mnt_best = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_log = self._train_epoch()
            val_log = evaluate(model=self.model, eval_loader=self.val_loader, device=self.device,
                               num_classes=self.val_loader.dataset.num_classes, criterion=self.criterion,
                               precision=self.precision, print_detail=False)
            self.scheduler.step(val_log['loss'])
            log = {'epoch': epoch,
                   'train': train_log,
                   'val':  val_log}
            print(f"Epoch :{epoch} \n Train : {log['train']['miou']} \n {log['val']['miou']}")
            self.improved = (val_log[self.metric] > self.mnt_best)


            if self.improved:
                self.mnt_best = val_log[self.metric]
                self.not_improved_count = 0
            else:
                self.not_improved_count += 1
                if self.early_stoping is not None and self.not_improved_count > self.early_stoping:
                    print(f'\nPerformance didn\'t improve for {self.early_stoping} epochs')
                    print('Training Stoped')
                    break
            # SAVE CHECKPOINT
            if self.improved:
                self._save_checkpoint(epoch, save_best=self.improved)

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.save_dir, f'checkpoint-epoch{epoch}.pth')
        print(f'\nSaving a checkpoint: {filename} ...')
        torch.save(state, filename)

        if save_best:
            filename = os.path.join(self.save_dir, f'best_model.pth')
            torch.save(state, filename)
            print("Saving current best: best_model.pth")


    def _resume_checkpoint(self, resume_path):
        self.logger.info(f'Loading checkpoint : {resume_path}')
        checkpoint = torch.load(resume_path)

        # Load last run info, the model params, the optimizer and the loggers
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.not_improved_count = 0
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print(f'Checkpoint <{resume_path}> (epoch {self.start_epoch}) was loaded')
